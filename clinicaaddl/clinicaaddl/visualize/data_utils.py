import sys, os

sys.path.insert(0, os.path.abspath('./'))

from tools.deep_learning.cnn_utils import evaluate_prediction

possible_selection_metrics = ["best_loss", "best_balanced_accuracy", "last_checkpoint"]


def get_results(args, average_fold=True):
    import pandas as pd
    import os
    import pathlib
    import numpy as np

    
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
                if "test" in mode and args.get_test_from_bayesian and args.bayesian:

                    values_df = stat_dict[fold][selection_metric].groupby("mode")
                    values_df = values_df.get_group(mode)
                    prediction_column = "predicted_label_from_%s" % args.ba_inference_mode
                    test_diagnosis_dict = evaluate_prediction(values_df[["true_label"]].values.astype(int),
                                                              values_df[[prediction_column]].values.astype(int))
                    test_diagnosis_df = pd.DataFrame(test_diagnosis_dict, index=[0])
                    test_diagnosis_df = test_diagnosis_df.assign(fold=fold,
                                                                 mode=mode)
                    results_dict[selection_metric] = pd.concat([results_dict[selection_metric], test_diagnosis_df],
                                                           axis=0)

                else:
                    test_diagnosis_path = os.path.join(cnn_classification_dir, selection_metric,
                                                       '%s_image_level_metrics.tsv' % (mode))
                    if os.path.exists(test_diagnosis_path):
                        test_diagnosis_df = pd.read_csv(test_diagnosis_path, sep='\t')
                        test_diagnosis_df = test_diagnosis_df.assign(fold=fold,
                                                                 mode=mode)

                        results_dict[selection_metric] = pd.concat([results_dict[selection_metric], test_diagnosis_df],
                                                           axis=0)

    resulting_metrics_dict = {}
    if average_fold:
        for key in results_dict.keys():
            res_df = results_dict[key].drop(
                ["total_loss", "image_id", "fold"], axis=1)
            resulting_metrics_dict[key] = res_df.groupby(["mode"], as_index=False, sort=False).agg(np.mean)
        resulting_metrics_dict = {"folds_average": resulting_metrics_dict}

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
            stat_df = stat_dict[key]
            additional_colums_df = stat_df[
                ["true_label", "predicted_label_from_mean", "predicted_label_from_mode", "mode", "participant_id"]]
            additional_colums_df = additional_colums_df.groupby(["mode", "participant_id"], as_index=False,
                                                                sort=False).agg(pd.Series.mode)
            stat_df = stat_df.drop(
                ["true_label", "predicted_label_from_mean", "predicted_label_from_mode", "fold"], axis=1)
            resulting_stat_dict[key] = stat_df.groupby(["mode", "participant_id"], as_index=False, sort=False).agg(np.mean)
            resulting_stat_dict[key]=resulting_stat_dict[key].merge(additional_colums_df, on=["mode", "participant_id"], how="right")
        resulting_stat_dict = {"folds_average": resulting_stat_dict}
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
        history_df = {"folds_average": history_df.groupby("epoch", as_index=False).agg(np.mean)}
    else:
        history_df = dict(list(history_df.groupby("fold")))
    return history_df

def reshape_dictionary(dict_sample):
    res = dict()
    for key, val in dict_sample.items():
        for key_in, val_in in val.items():
            if key_in not in res:
                temp = dict()
            else:
                temp = res[key_in]
            temp[key] = val_in
            res[key_in] = temp
    return res



def get_data_generic(args, reshape_dict=True):
    data = {}
    for data_type in args.data_types:
        data[data_type] = eval("get_%s" % data_type)(args, args.average_fold)
    #data is now in format {data_type: {fold_0:, ...fold_n etc}}
    if reshape_dict:
        # reshape data to format  {fold_0: {data_type_1:, ...data_type_i etc}}
        data = reshape_dictionary(data)
    return data
