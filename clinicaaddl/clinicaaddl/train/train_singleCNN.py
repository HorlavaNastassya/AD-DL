# coding: utf8

import os
import torch
from torch.utils.data import DataLoader


import sys, os
sys.path.insert(0, os.path.abspath('./'))

from tools.deep_learning.models import transfer_learning, init_model, load_model, load_optimizer
from tools.deep_learning.data import (get_transforms,
                                        load_data,
                                        return_dataset,
                                        generate_sampler)
from tools.deep_learning.cnn_utils import train, get_criterion, test, mode_level_to_tsvs, soft_voting_to_tsvs, get_classWeights
from tools.deep_learning.iotools import return_logger, check_and_clean
from tools.deep_learning.iotools import commandline_to_json, write_requirements_version, translate_parameters


def train_single_cnn(params):
    """
    Trains a single CNN and writes:
        - logs obtained with Tensorboard during training,
        - best models obtained according to two metrics on the validation set (loss and balanced accuracy),
        - for patch and roi modes, the initialization state is saved as it is identical across all folds,
        - final performances at the end of the training.

    If the training crashes it is possible to relaunch the training process from the checkpoint.pth.tar and
    optimizer.pth.tar files which respectively contains the state of the model and the optimizer at the end
    of the last epoch that was completed before the crash.
    """
    main_logger = return_logger(params.verbose, "main process")
    train_logger = return_logger(params.verbose, "train")
    eval_logger = return_logger(params.verbose, "final evaluation")

    # params.output_dir=check_and_clean(params.output_dir)
    #
    # commandline_to_json(params, logger=main_logger)

    # write_requirements_version(params.output_dir)
    params = translate_parameters(params)
    train_transforms, all_transforms = get_transforms(params.mode,
                                                      minmaxnormalization=params.minmaxnormalization,
                                                      data_augmentation=params.data_augmentation,
                                                      output_dir=params.output_dir)

    if params.split is None:
        if params.n_splits is None:
            fold_iterator = range(1)
        else:
            fold_iterator = range(params.n_splits)
    else:
        fold_iterator = params.split

    for fi in fold_iterator:
        main_logger.info("Fold %i" % fi)
        # Initialize the model
        main_logger.info('Initialization of the model')

        # Initialize the model
        print('Initialization of the model')
        model = init_model(params, initial_shape=None)

        # Define output directories
        log_dir = os.path.join(
            params.output_dir, 'fold-%i' % fi, 'tensorboard_logs')
        fold_dir = os.path.join(
            params.output_dir, 'fold-%i' % fi)
        model_dir = os.path.join(fold_dir, 'models')


        training_df, valid_df = load_data(
            params.tsv_path,
            params.diagnoses,
            fi,
            n_splits=params.n_splits,
            baseline=params.baseline,
            logger=main_logger
        )

        data_train = return_dataset(params.mode, params.input_dir, training_df, params.preprocessing,
                                    train_transformations=train_transforms, all_transformations=all_transforms,
                                    params=params)
        data_valid = return_dataset(params.mode, params.input_dir, valid_df, params.preprocessing,
                                    train_transformations=train_transforms, all_transformations=all_transforms,
                                    params=params)

        train_sampler = generate_sampler(data_train, params.sampler)

        train_loader = DataLoader(
            data_train,
            batch_size=params.batch_size,
            sampler=train_sampler,
            num_workers=params.num_workers,
            pin_memory=True
        )

        valid_loader = DataLoader(
            data_valid,
            batch_size=params.batch_size,
            shuffle=False,
            num_workers=params.num_workers,
            pin_memory=True
        )

        # Define criterion and optimizer

        normedWeights = get_classWeights(params, training_df)
        criterion = get_criterion(params.loss, normedWeights)
        #         optimizer = getattr(torch.optim, params.optimizer)(filter(lambda x: x.requires_grad, model.parameters()),
        #                                                            lr=params.learning_rate,
        #                                                            weight_decay=params.weight_decay)
        resume_fold = params.resume
        if params.resume:
            if os.path.exists(os.path.join(model_dir,'checkpoint.pth.tar')) and os.path.exists(os.path.join(model_dir,'optimizer.pth.tar')):
                model, beginning_epoch = load_model(model, model_dir, params.gpu, 'checkpoint.pth.tar')
                optimizer_path = os.path.join(model_dir, 'optimizer.pth.tar')
                optimizer, optimizer_epoch = load_optimizer(optimizer_path, model, params.gpu)

                if beginning_epoch != optimizer_epoch:
                    print('!!! Last model epoch and last optimizer_epoch does not match!!!')
                params.beginning_epoch = beginning_epoch + 1

            else:
                resume_fold = False

        if not resume_fold:
            model = transfer_learning(model, fi, source_path=params.transfer_learning_path,
                                      gpu=params.gpu, selection=params.transfer_learning_selection,
                                      logger=main_logger)
            optimizer = getattr(torch.optim, params.optimizer)(filter(lambda x: x.requires_grad, model.parameters()),
                                                               lr=params.learning_rate,
                                                               weight_decay=params.weight_decay)
            params.beginning_epoch=0

        main_logger.debug('Beginning the training task')
        # toDO: resume as argument from command line
        if params.beginning_epoch<params.epochs:
            train(model, train_loader, valid_loader, criterion,
                  optimizer, resume_fold, log_dir, model_dir, params, train_logger)

        test_single_cnn(model, params.output_dir, train_loader, "train",
                            fi, criterion, params.mode, eval_logger, params.selection_threshold, gpu=params.gpu, skip_if_exist=True)
        test_single_cnn(model, params.output_dir, valid_loader, "validation",
                            fi, criterion, params.mode, eval_logger, params.selection_threshold, gpu=params.gpu,  skip_if_exist=True)


def test_single_cnn(model, output_dir, data_loader, subset_name, split, criterion, mode, logger, selection_threshold,
                    gpu=False, skip_if_exist=False):

    for selection in ["best_balanced_accuracy", "best_loss", "last_checkpoint"]:
        if skip_if_exist:
            if not os.path.exists(os.path.join(output_dir, 'fold-%i' % split, 'cnn_classification', selection, '%s_%s_level_prediction.tsv' % (subset_name, mode))):
                # load the best trained model during the training
                
                if selection == "last_checkpoint":
                    full_model_path = os.path.join(output_dir, 'fold-%i' % split, 'models')
                    model_filename = 'checkpoint.pth.tar'
                else:
                    full_model_path = os.path.join(output_dir, 'fold-%i' % split, 'models', selection)
                    model_filename = 'model_best.pth.tar'
                    
                model, best_epoch = load_model(model, full_model_path,
                                               gpu=gpu, filename=model_filename)

                results_df, metrics = test(model, data_loader, gpu, criterion, mode)
                logger.info("%s level %s balanced accuracy is %f for model selected on %s"
                            % (mode, subset_name, metrics["balanced_accuracy"], selection))

                mode_level_to_tsvs(output_dir, results_df, metrics, split, selection, mode, dataset=subset_name)

                # Soft voting
                if mode in ["patch", "roi", "slice"]:
                    soft_voting_to_tsvs(output_dir, split, logger=logger, selection=selection, mode=mode,
                                        dataset=subset_name, selection_threshold=selection_threshold)
