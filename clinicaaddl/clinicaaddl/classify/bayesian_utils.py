import os
from os.path import isdir, join, abspath, exists

import sys, os
sys.path.insert(0, os.path.abspath('./'))
from tools.deep_learning.iotools import return_logger
from tools.deep_learning import read_json
from tools.deep_learning.cnn_utils import evaluate_prediction, mode_level_to_tsvs

import pathlib
from os import strerror
import errno


def bayesian_predictions(
        model_path,
        prefixes,
        verbose=0,
        function=None,
        **kwards
):
    # prefix: e.g. test_1.5T
    # selection_metrics: best_loss/best accuracy etc
    import argparse
    possible_selection_metrics = ["best_loss", "best_balanced_accuracy", "last_checkpoint"]
    logger = return_logger(verbose, "classify")
    predictions_path = abspath(model_path)

    parser = argparse.ArgumentParser()
    parser.add_argument("model_path", type=str,
                        help="Path to the trained model folder.")
    options = parser.parse_args([predictions_path])
    json_file = join(predictions_path, 'commandline.json')
    options = read_json(options, json_path=json_file)

    if not isdir(predictions_path):
        logger.error("A valid model in the path was not found. ")
        raise FileNotFoundError(
            errno.ENOENT, strerror(errno.ENOENT), predictions_path)

    # Define the path
    currentDirectory = pathlib.Path(predictions_path)
    # Search for 'fold-*' pattern
    currentPattern = "fold-*"

    # loop depending the number of folds found in the model folder
    for fold_dir in currentDirectory.glob(currentPattern):
        fold = int(str(fold_dir).split("-")[-1])
        fold_path = join(predictions_path, fold_dir)
        selection_metrics = []
        cnn_classification_dir = os.path.join(fold_path, 'cnn_classification')
        for f in os.scandir(cnn_classification_dir):
            if os.path.basename(os.path.normpath(f.path)) in possible_selection_metrics:
                selection_metrics.append(os.path.basename(os.path.normpath(f.path)))

        predictions_path = join(cnn_classification_dir,'bayesian_predictions')
#         print("Model:%s" % predictions_path)

        for selection_metric in selection_metrics:
            for prefix in prefixes:
                full_predictions_path = join(predictions_path, selection_metric, prefix)
                args = {"model_path": full_predictions_path,
                        "prefix": prefix,
                        "output_path": currentDirectory,
                        "fold": fold,
                        "selection": selection_metric,
                        "mode": options.mode,
                        }
                if function == "inference":
                    args["from_mean"] = kwards["from_mean"]

                getattr(BayesianFunctionality, function)(**args)
        # print("__________________________________________________________________")


class BayesianFunctionality():
    def __init__(self):
        self.functions = ["inference_from_mode", "inference_from_mean", "uncertainty_metrics"]

    @staticmethod
    def inference(
            predictions_path,
            prefix,
            output_dir,
            fold,
            selection,
            mode='image',
            from_mean=True,
            get_previous_loss=True,
    ):
        import pandas as pd
        import numpy as np
        import scipy.stats as st

        columns = ["participant_id", "session_id", "true_label", "predicted_label"]
        results_df = pd.DataFrame(columns=columns)

        performance_dir = os.path.join(output_dir, 'fold-%i' % fold, 'cnn_classification', selection)
        metrics_filename = os.path.join(performance_dir, '%s_%s_level_metrics.tsv' % (prefix, mode))
        old_results_df = pd.read_csv(metrics_filename, '\t')

        predictions_dict = get_bayesian_prediction_from_model_generic(predictions_path)

        for id in predictions_dict.keys():
            if from_mean:
                mean_predicted_class = np.mean(predictions_dict[id]["bayesian_predictions"], axis=0)
                predicted_class = np.argmax(mean_predicted_class, axis=-1)
            else:
                predicted_class, _ = st.mode(np.argmax(predictions_dict[id]["bayesian_predictions"], axis=1), axis=0)
                predicted_class = predicted_class[0]

            row = [[id, predictions_dict[id]["session_id"], predictions_dict[id]["true_label"], predicted_class]]
            row_df = pd.DataFrame(row, columns=columns)
            results_df = pd.concat([results_df, row_df])

            results_df.reset_index(inplace=True, drop=True)

        metrics_dict = evaluate_prediction(results_df.true_label.values.astype(int),
                                           results_df.predicted_label.values.astype(int))
        if get_previous_loss:
            metrics_dict['total_loss'] = old_results_df["total_loss"][0]

        prefix = prefix + "_from-mean" if from_mean else prefix
        mode_level_to_tsvs(output_dir, results_df, metrics_dict, fold, selection, mode,
                           dataset=prefix)

    @staticmethod
    def stat(model_path,
             output_path,
             prefix,
            fold,
            selection,
            mode='image',
    ):
        import pandas as pd
        
        import scipy.stats as st
        import numpy as np

        columns = ["participant_id", "session_id", "true_label", "predicted_label_from_mean", "predicted_label_from_mode", "class_variance", "total_variance", "entropy", "NLL"]
        # Saving following metrics: class_variance: variance for each class(column) separately [12 value per object];
        # total_variance: sum of var over columns [1 value per object];
        # NLL=-log(highest probability of mean probabilities over T runs [1 value per object];
        # entropy: [1 value per object]

        results_df = pd.DataFrame(columns=columns)

        performance_dir = os.path.join(output_path, 'fold-%i' % fold, 'cnn_classification', selection)
        predictions_dict = get_bayesian_prediction_from_model_generic(model_path)
        stats_path = os.path.join(performance_dir, "bayesian_statistics")
        os.makedirs(stats_path, exist_ok=True)
        stats_filename=os.path.join(stats_path, '%s_%s_level_stats.tsv' % (prefix, mode))

        for id in predictions_dict.keys():
            mean_predicted_class = np.mean(predictions_dict[id]["bayesian_predictions"], axis=0)
            predicted_class_from_mean = np.argmax(mean_predicted_class, axis=-1)
            
            predicted_class_from_mode, _ = st.mode(np.argmax(predictions_dict[id]["bayesian_predictions"], axis=1), axis=0)
            predicted_class_from_mode = predicted_class_from_mode[0]
                
            #Note: nll is taking into consideration prediction from mean, not from the mode
            nll_row=-np.log(mean_predicted_class[predicted_class_from_mean])
            class_variance_row=np.var(predictions_dict[id]["bayesian_predictions"], axis=0)
            total_variance_row=np.sum(class_variance_row)
            entropy_row=-np.sum(mean_predicted_class*np.log2(mean_predicted_class+1E-14))
            
            

            row = [[id, predictions_dict[id]["session_id"], predictions_dict[id]["true_label"],predicted_class_from_mean, predicted_class_from_mode, class_variance_row, total_variance_row, entropy_row, nll_row ]]
            row_df = pd.DataFrame(row, columns=columns)
            results_df = pd.concat([results_df, row_df])
            results_df.reset_index(inplace=True, drop=True)

        results_df.to_csv(stats_filename, index=False, sep='\t')


def get_bayesian_prediction_from_model_generic(predictions_path):
    import pandas as pd
    modelPatter = "subject_model*"
    bayes_prediction_files = [f for f in pathlib.Path(predictions_path).glob("test_*")]
    predictions_dict = {}
    for idx, prediction_file in enumerate(bayes_prediction_files):
        bayes_predictions_df = pd.read_csv(prediction_file, '\t')
        for index, row in bayes_predictions_df.iterrows():
            if not row["participant_id"] in predictions_dict.keys():
                predictions_dict[row["participant_id"]] = {'session_id': None, "bayesian_predictions": [],
                                                           "true_label": None}
            predicted_column_names = []
            for s in bayes_predictions_df.columns:
                if "predicted_column" in s:
                    predicted_column_names.append(s)

            predictions_dict[row["participant_id"]]["bayesian_predictions"].append(
                [row[s] for s in predicted_column_names])

            if predictions_dict[row["participant_id"]]["true_label"] is None:
                predictions_dict[row["participant_id"]]["true_label"] = row["true_label"]
            if predictions_dict[row["participant_id"]]["session_id"] is None:
                predictions_dict[row["participant_id"]]["session_id"] = row["session_id"]
    return predictions_dict