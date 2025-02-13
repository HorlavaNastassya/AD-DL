# coding: utf8

from os.path import isdir, join, abspath, exists
from os import strerror, makedirs, listdir
import errno
import pathlib

import sys, os
sys.path.insert(0, os.path.abspath('./'))
from tools.deep_learning import create_model, load_model, read_json
from tools.deep_learning.iotools import return_logger, translate_parameters
from tools.deep_learning.data import return_dataset, get_transforms, compute_num_cnn, load_data_test
from tools.deep_learning.cnn_utils import test, test_bayesian, soft_voting_to_tsvs, mode_level_to_tsvs, get_criterion, get_classWeights, bayesian_predicions_to_tsvs
import torch.nn as nn
from torch.utils.data import DataLoader



def classify(caps_dir,
             tsv_path,
             model_path,
             prefix_output,
             labels=True,
             gpu=True,
             num_workers=0,
             batch_size=1,
             prepare_dl=True,
             selection_metrics=None,
             diagnoses=None,
             verbose=0,
             baseline=True,
             bayesian=False,
             nbr_bayesian_iter=None
             ):
    """
    This function verifies the input folders, and the existence of the json file
    then it launch the inference stage from a specific model.

    It writes the outputs in <model_path>/fold-<fold>/cnn_classification.

    Args:
        caps_dir: folder containing the tensor files (.pt version of MRI)
        tsv_path: file with the name of the MRIs to process (single or multiple)
        model_path: file with the model (pth format).
        prefix_output: prefix of all classification outputs.
        labels: by default is True. If False no metrics tsv files will be written.
        gpu: if true, it uses gpu.
        num_workers: num_workers used in DataLoader
        batch_size: batch size of the DataLoader
        prepare_dl: if true, uses extracted patches/slices otherwise extract them
        on-the-fly.
        selection_metrics: list of metrics to find best models to be evaluated.
        diagnoses: list of diagnoses to be tested if tsv_path is a folder.
        verbose: level of verbosity.

    """
    logger = return_logger(verbose, "classify")

    # Verify that paths exist
    caps_dir = abspath(caps_dir)
    model_path = abspath(model_path)
    tsv_path = abspath(tsv_path)

    if not isdir(caps_dir):
        logger.error("Folder containing MRIs was not found, please verify its location.")
        raise FileNotFoundError(
            errno.ENOENT, strerror(errno.ENOENT), caps_dir)
    if not isdir(model_path):
        logger.error("A valid model in the path was not found. Donwload them from aramislab.inria.fr")
        raise FileNotFoundError(
            errno.ENOENT, strerror(errno.ENOENT), model_path)
    if not exists(tsv_path):
        raise FileNotFoundError(
            errno.ENOENT, strerror(errno.ENOENT), tsv_path)

    # Infer json file from model_path (suppose that json file is at the same
    # folder)

    json_file = join(model_path, 'commandline.json')

    if not exists(json_file):
        logger.error("Json file doesn't exist")
        raise FileNotFoundError(
            errno.ENOENT, strerror(errno.ENOENT), json_file)

    inference_from_model(
        caps_dir,
        tsv_path,
        model_path,
        json_file,
        prefix_output,
        labels,
        gpu,
        num_workers,
        batch_size,
        prepare_dl,
        selection_metrics,
        diagnoses,
        logger,
        baseline,
        bayesian,
        nbr_bayesian_iter
    )


def inference_from_model(caps_dir,
                         tsv_path,
                         model_path=None,
                         json_file=None,
                         prefix=None,
                         labels=True,
                         gpu=True,
                         num_workers=0,
                         batch_size=1,
                         prepare_dl=False,
                         selection_metrics=None,
                         diagnoses=None,
                         logger=None,
                         baseline=True,
                         bayesian=False,
                         nbr_bayesian_iter=None
                         ):
    """
    Inference from previously trained model.

    This functions uses a previously trained model to classify the input(s).
    The model is stored in the variable model_path and it assumes the folder
    structure given by the training stage. Particullary to have a prediction at
    image level, it assumes that results of the validation set are stored in
    the model_path folder in order to perform soft-voiting at the slice/patch
    level and also for multicnn.

    Args:
        caps_dir: folder containing the tensor files (.pt version of MRI)
        tsv_path: file with the name of the MRIs to process (single or multiple)
        model_path: file with the model (pth format).
        json_file: file containing the training parameters.
        prefix: prefix of all classification outputs.
        labels: by default is True. If False no metrics tsv files will be written.
        measurements.tsv
        gpu: if true, it uses gpu.
        num_workers: num_workers used in DataLoader
        batch_size: batch size of the DataLoader
        prepare_dl: if true, uses extracted patches/slices otherwise extract them
        on-the-fly.
        selection_metrics: list of metrics to find best models to be evaluated.
        diagnoses: list of diagnoses to be tested if tsv_path is a folder.
        logger: Logger instance.

    Returns:
        Files written in the output folder with prediction results and metrics. By
        default the output folder is named cnn_classification and it is inside the
        model_folder.

    Raises:


    """
    import argparse
    import logging

    if logger is None:
        logger = logging

    parser = argparse.ArgumentParser()
    parser.add_argument("model_path", type=str,
                        help="Path to the trained model folder.")
    options = parser.parse_args([model_path])
    options = read_json(options, json_path=json_file)

    logger.debug("Load model with these options:")
    logger.debug(options)

    # Overwrite options with user input
    options.use_cpu = not gpu
    options.nproc = num_workers
    options.batch_size = batch_size
    options.prepare_dl = prepare_dl
    if diagnoses is not None:
        options.diagnoses = diagnoses

    options = translate_parameters(options)

    if options.mode_task == "multicnn":
        num_cnn = compute_num_cnn(caps_dir, tsv_path, options, "test")
    else:
        num_cnn = None
    # Define the path
    currentDirectory = pathlib.Path(model_path)
    # Search for 'fold-*' pattern
    currentPattern = "fold-*"

    # loop depending the number of folds found in the model folder
    for fold_dir in currentDirectory.glob(currentPattern):
        fold = int(str(fold_dir).split("-")[-1])
        fold_path = join(model_path, fold_dir)
        model_path = join(fold_path, 'models')

        for selection_metric in selection_metrics:

            if options.mode_task == 'multicnn':
                for cnn_dir in listdir(model_path):
                    if not exists(join(model_path, cnn_dir, "best_%s" % selection_metric, 'model_best.pth.tar')):
                        raise FileNotFoundError(
                            errno.ENOENT,
                            strerror(errno.ENOENT),
                            join(model_path,
                                 cnn_dir,
                                 "best_%s" % selection_metric,
                                 'model_best.pth.tar')
                        )

            else:
                if selection_metric=="last_checkpoint":
                    full_model_path = model_path
                    name='checkpoint.pth.tar'
                else:
                    full_model_path = join(model_path, "best_%s" % selection_metric)
                    name='model_best.pth.tar'
                if not exists(join(full_model_path, name)):
                    raise FileNotFoundError(
                        errno.ENOENT,
                        strerror(errno.ENOENT),
                        join(full_model_path, name))
            mode_prefix=selection_metric if selection_metric=="last_checkpoint" else 'best_%s' % selection_metric

            performance_dir = join(fold_path, 'cnn_classification', mode_prefix)

            makedirs(performance_dir, exist_ok=True)

            if bayesian != options.bayesian:
                print("You provided the network type as bayesian %str, but your network was trained as bayesian %str"%(bayesian, options.bayesian))
                print("Algoritm will use the parameters used during training")
                bayesian=options.bayesian
            # It launch the corresponding function, depending on the mode.
            inference_from_model_generic(
                caps_dir,
                tsv_path,
                model_path,
                options,
                prefix,
                currentDirectory,
                fold,
                mode_prefix,
                labels=labels,
                num_cnn=num_cnn,
                logger=logger,
                baseline=baseline,
                bayesian=bayesian,
                nbr_bayesian_iter=nbr_bayesian_iter
            )

            # Soft voting
            if hasattr(options, 'selection_threshold'):
                selection_thresh = options.selection_threshold
            else:
                selection_thresh = 0.8

            # Write files at the image level (for patch, roi and slice).
            # It assumes the existance of validation files to perform soft-voting
            if options.mode in ["patch", "roi", "slice"]:
                soft_voting_to_tsvs(currentDirectory, fold, mode_prefix, options.mode,
                                    prefix, num_cnn=num_cnn, selection_threshold=selection_thresh,
                                    use_labels=labels, logger=logger)

            logger.info("Prediction results and metrics are written in the "
                        "following folder: %s" % performance_dir)


def inference_from_model_generic(caps_dir, tsv_path, model_path, model_options,
                                 prefix, output_dir, fold, selection,
                                 labels=True, num_cnn=None, logger=None, baseline=True,
                                 bayesian=False, nbr_bayesian_iter=None
                                 ):
    from os.path import join
    import logging


    if logger is None:
        logger = logging

    gpu = not model_options.use_cpu

    _, all_transforms = get_transforms(model_options.mode, model_options.minmaxnormalization)

    test_df = load_data_test(tsv_path, model_options.diagnoses, baseline)

    # Define loss and optimizer
    normedWeights = get_classWeights(model_options, test_df)
    criterion = get_criterion(model_options.loss, normedWeights)

    if model_options.mode_task == 'multicnn':

        for n in range(num_cnn):

            test_dataset = return_dataset(
                model_options.mode,
                caps_dir,
                test_df,
                model_options.preprocessing,
                train_transformations=None,
                all_transformations=all_transforms,
                params=model_options,
                cnn_index=n,
                labels=labels
            )

            test_loader = DataLoader(
                test_dataset,
                batch_size=model_options.batch_size,
                shuffle=False,
                num_workers=model_options.nproc,
                pin_memory=True)

            # load the best trained model during the training
            model = create_model(model_options, test_dataset.size)

            if selection == "last_checkpoint":
                full_model_path = join(model_path, 'cnn-%i' % n)
                filename = 'checkpoint.pth.tar'
            else:
                full_model_path = join(model_path, 'cnn-%i' % n, selection)
                filename = 'model_best.pth.tar'

            model, best_epoch = load_model(
                model,
                full_model_path,
                gpu,
                filename=filename)

            cnn_df, cnn_metrics = test(
                model,
                test_loader,
                gpu,
                criterion,
                mode=model_options.mode,
                use_labels=labels
            )

            if labels:
                logger.info("%s balanced accuracy is %f for %s %i and model selected on %s"
                            % (prefix, cnn_metrics["balanced_accuracy"], model_options.mode, n, selection))

            mode_level_to_tsvs(output_dir, cnn_df, cnn_metrics, fold, selection, model_options.mode,
                               dataset=prefix, cnn_index=n)

    else:

        # Read/localize the data
        test_dataset = return_dataset(
            model_options.mode,
            caps_dir,
            test_df,
            model_options.preprocessing,
            train_transformations=None,
            all_transformations=all_transforms,
            params=model_options,
            labels=labels
        )

        # Load the data
        test_loader = DataLoader(
            test_dataset,
            batch_size=model_options.batch_size,
            shuffle=False,
            num_workers=model_options.nproc,
            pin_memory=True)

        # Load model from path
        model = create_model(model_options, test_dataset.size)

        if selection == "last_checkpoint":
            full_model_path = model_path
            filename = 'checkpoint.pth.tar'
        else:
            full_model_path = join(model_path, selection)
            filename = 'model_best.pth.tar'

        best_model, best_epoch = load_model(
            model, full_model_path,
            gpu, filename=filename)

        # Run the model on the data
        if bayesian:

            predictions_df, metrics, bayesian_predictions_list = test_bayesian(
                best_model,
                test_loader,
                gpu,
                criterion,
                use_labels=labels,
                nbr_bayesian_iter=nbr_bayesian_iter,
            )

            bayesian_predicions_to_tsvs(output_dir, bayesian_predictions_list, fold, selection, model_options.mode,
                               dataset=prefix)

        else:
            predictions_df, metrics = test(
                best_model,
                test_loader,
                gpu,
                criterion,
                mode=model_options.mode,
                use_labels=labels
            )

        if labels:
            logger.info("%s level %s balanced accuracy is %f for model selected on %s"
                        % (model_options.mode, prefix, metrics["balanced_accuracy"], selection))

        mode_level_to_tsvs(output_dir, predictions_df, metrics, fold, selection, model_options.mode,
                           dataset=prefix)


