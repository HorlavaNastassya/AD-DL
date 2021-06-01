# coding: utf8

import argparse
from os import path
from time import time
import torch
from torch.utils.data import DataLoader

from .train_singleCNN import test_single_cnn
from clinicaaddl.tools.deep_learning.data import MRIDataset, MinMaxNormalization, load_data
from clinicaaddl.tools.deep_learning import create_model, load_model, load_optimizer, read_json
from clinicaaddl.tools.deep_learning.cnn_utils import train

# parser = argparse.ArgumentParser(description="Argparser for Pytorch 3D CNN")
#
# # Mandatory arguments
# parser.add_argument("model_path", type=str,
#                     help="model selected")
# parser.add_argument("split", type=int,
#                     help="Will load the specific split wanted.")
#
# # Computational argument
# parser.add_argument('--gpu', action='store_true', default=False,
#                     help='Uses gpu instead of cpu if cuda is available')
# parser.add_argument("--num_workers", '-w', default=1, type=int,
#                     help='the number of batch being loaded in parallel')
    
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
                                                      data_augmentation=params.data_augmentation, output_dir=params.output_dir)

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
        fold_dir= os.path.join(
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

        normedWeights=get_classWeights(params, training_df)
        criterion = get_criterion(params.loss, normedWeights)
#         optimizer = getattr(torch.optim, params.optimizer)(filter(lambda x: x.requires_grad, model.parameters()),
#                                                            lr=params.learning_rate,
#                                                            weight_decay=params.weight_decay)

        if os.path.exists(fold_dir):
            model, params.beginning_epoch = load_model(model, model_dir, options.gpu, 'checkpoint.pth.tar')
            optimizer_path = path.join(options.model_path, 'optimizer.pth.tar')
            optimizer = load_optimizer(optimizer_path, model)
            resume_flag=True
        else:
            model = init_model(params, initial_shape=None)
            model = transfer_learning(model, fi, source_path=params.transfer_learning_path,
                                      gpu=params.gpu, selection=params.transfer_learning_selection,
                                      logger=main_logger)
            optimizer = getattr(torch.optim, params.optimizer)(filter(lambda x: x.requires_grad, model.parameters()),
                                                           lr=params.learning_rate,
                                                           weight_decay=params.weight_decay)
            resume_flag=False

        main_logger.debug('Beginning the training task')
        #toDO: resume as argument from command line
        train(model, train_loader, valid_loader, criterion,
              optimizer, resume_flag, log_dir, model_dir, params, train_logger)

        test_single_cnn(model, params.output_dir, train_loader, "train",
                        fi, criterion, params.mode, eval_logger, params.selection_threshold, gpu=params.gpu)
        test_single_cnn(model, params.output_dir, valid_loader, "validation",
                        fi, criterion, params.mode, eval_logger, params.selection_threshold, gpu=params.gpu)

if __name__ == "__main__":
    commandline = parser.parse_known_args()
    options = commandline[0]
    if commandline[1]:
        print("unknown arguments: %s" % parser.parse_known_args()[1])
    main(options)
