import sys, os

sys.path.insert(0, os.path.abspath('./'))

from tools.deep_learning.cnn_utils import evaluate_prediction

possible_selection_metrics = ["best_loss", "best_balanced_accuracy", "last_checkpoint"]


def get_results(args, average_fold=True):
    import pandas as pd
    import os
    import pathlib
    import numpy as np

    if args.metrics is None:
        metrics = ["accuracy", 'sensitivity', 'precision', 'f1-score']
    else:
        metrics = args.metrics

    if args.get_test_from_bayesian:
        stat_dict = get_uncertainty_distribution(args, average_fold=False)

    results_dict = {}
    currentDirectory = pathlib.Path(args.model_path)
    currentPattern = "fold-*"

    for fold_dir in currentDirectory.glob(currentPattern):
        fold = int(str(fold_dir).split("-")[-1])

        selection_metrics = []
        cnn_classification_dir = os.path.join(args.model_path, 'fold-%i' % fold, 'cnn_classification')
        for f in os.scandir(cnn_classification_dir):
            if os.path.basename(os.path.normpath(f.path)) in possible_selection_metrics:
                selection_metrics.append(os.path.basename(os.path.normpath(f.path)))

        for selection_metric in selection_metrics:
            if not selection_metric in results_dict.keys():
                results_dict[selection_metric] = pd.DataFrame()
            modes = ['train', 'validation']
            for ms_el in args.MS_list:
                modes.append('test_' + ms_el)

            for mode in modes:
                if "test" in mode and args.get_test_from_bayesian:

                    # toDo: read from bayes stat
                    values_df = stat_dict[fold][selection_metric].groupby("mode")
                    values_df = values_df.get_group(mode)
                    prediction_column = "predicted_label_from_%s" % args.ba_inference_mode
                    test_diagnosis_dict = evaluate_prediction(values_df.true_label.values.astype(int),
                                                              values_df[[prediction_column]].values.astype(int))
                    test_diagnosis_df = pd.DataFrame(test_diagnosis_dict, index=[0])
                    test_diagnosis_df = test_diagnosis_df.assign(fold=fold,
                                                                 mode=mode)

                else:
                    test_diagnosis_path = os.path.join(cnn_classification_dir, selection_metric,
                                                       '%s_image_level_metrics.tsv' % (mode))
                    test_diagnosis_df = pd.read_csv(test_diagnosis_path, sep='\t')
                    test_diagnosis_df = test_diagnosis_df.assign(fold=fold,
                                                                 mode=mode)

                results_dict[selection_metric] = pd.concat([results_dict[selection_metric], test_diagnosis_df],
                                                           axis=0)

    resulting_metrics_dict = {}
    if average_fold:
        for key in results_dict.keys():
            res_df=results_dict[key].drop(
                ["total_loss", "image_id"], axis=1)
            resulting_metrics_dict[key] = res_df.groupby(["mode"], as_index=False).agg(np.mean)
    else:
        for key in results_dict.keys():
            res_df = results_dict[key].drop(["total_loss", "image_id"], axis=1)
            metric_dict = dict(list(res_df.groupby("fold")))
            for fold in metric_dict.keys():
                if fold not in resulting_metrics_dict.keys():
                    resulting_metrics_dict[fold] = {}
                resulting_metrics_dict[fold][key] = metric_dict[fold]

    return resulting_metrics_dict


def get_uncertainty_distribution(args, average_fold=True):
    import pandas as pd
    import os
    import pathlib
    import numpy as np

    currentDirectory = pathlib.Path(args.model_path)
    currentPattern = "fold-*"
    # stat_df = pd.DataFrame( columns = ["fold", "selection_metric", "prefix",  "participant_id", "session_id", "true_label", "predicted_label_from_mean", "predicted_label_from_mode", "class_variance", "total_variance", "entropy", "NLL"])
    stat_dict = {}
    for fold_dir in currentDirectory.glob(currentPattern):
        fold = int(str(fold_dir).split("-")[-1])

        selection_metrics = []
        cnn_classification_dir = os.path.join(args.model_path, 'fold-%i' % fold, 'cnn_classification')
        for f in os.scandir(cnn_classification_dir):
            if os.path.basename(os.path.normpath(f.path)) in possible_selection_metrics:
                selection_metrics.append(os.path.basename(os.path.normpath(f.path)))

        for selection_metric in selection_metrics:
            if not selection_metric in stat_dict.keys():
                stat_dict[selection_metric] = pd.DataFrame()
            modes = ['test_' + ms_el for ms_el in args.MS_list]

            for mode in modes:
                test_diagnosis_path = os.path.join(cnn_classification_dir, selection_metric,
                                                   "bayesian_statistics", '%s_image_level_stats.tsv' % (mode))
                test_diagnosis_df = pd.read_csv(test_diagnosis_path, sep='\t')
                test_diagnosis_df["class_variance"] = test_diagnosis_df["class_variance"].apply(
                    lambda x: x[1:-1].split()).apply(lambda x: [float(i) for i in x])
                test_diagnosis_df = test_diagnosis_df.assign(fold=fold, mode=mode)
                stat_dict[selection_metric] = pd.concat([stat_dict[selection_metric], test_diagnosis_df], axis=0)
                # stat_dict[selection_metric].reset_index(inplace=True, drop=True)

    resulting_stat_dict = {}
    if average_fold:
        for key in stat_dict.keys():
            stat_df = stat_dict[key].drop(
                ["true_label", "predicted_label_from_mean", "predicted_label_from_mode"], axis=1)
            resulting_stat_dict[key] = stat_df.groupby(["mode", "participant_id"], as_index=False).agg(np.mean)
    else:
        for key in stat_dict.keys():
            metric_dict = dict(list(stat_dict[key].groupby("fold")))
            for fold in metric_dict.keys():
                if fold not in resulting_stat_dict.keys():
                    resulting_stat_dict[fold] = {}
                resulting_stat_dict[fold][key] = metric_dict[fold]

    return resulting_stat_dict


def get_history(args, average_fold=True):
    import pandas as pd
    import os
    import pathlib
    import numpy as np

    currentDirectory = pathlib.Path(args.model_path)
    currentPattern = "fold-*"
    history_df = pd.DataFrame()
    for fold_dir in currentDirectory.glob(currentPattern):
        fold = int(str(fold_dir).split("-")[-1])
        history = pd.read_csv(os.path.join(args.model_path, 'fold-%i' % fold, 'training.tsv'), sep='\t')
        history = history.assign(fold=fold)
        history_df = pd.concat([history_df, history], axis=0)
    if average_fold:
        history_df = history_df[
            ["epoch", "balanced_accuracy_train", "loss_train", "balanced_accuracy_valid", "loss_valid"]]
        history_df = history_df.groupby("epoch").agg(np.mean)
    else:
        history_df = dict(list(history_df.groupby("fold")))
    return history_df


def get_data_generic(args):
    data = {}
    for data_type in args.data_types:
        data[data_type] = eval("get_%s" % data_type)(args, args.average_fold)
    return data