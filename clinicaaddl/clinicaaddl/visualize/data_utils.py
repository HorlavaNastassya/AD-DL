import sys, os

sys.path.insert(0, os.path.abspath('./'))

from tools.deep_learning.cnn_utils import evaluate_prediction

possible_selection_metrics = ["best_loss", "best_balanced_accuracy", "last_checkpoint"]

def get_test_prediction(test_diagnosis_df, merged_df):
    import pandas as pd

    test_diagnosis_df=merged_df.merge(test_diagnosis_df, on=["participant_id"], how="right")
    test_diagnosis_df=dict(list(test_diagnosis_df.groupby("T1w_mri_field")))
    res_dict={}
    for key in [1.5, 3]:
        res_dict["test_%sT"%str(key)]=test_diagnosis_df[key]
    return res_dict

def get_results(args, aggregation_type="average"):
    # aggregation_type=[average, separate, together]
    import pandas as pd
    import os
    import pathlib
    import numpy as np

    

    if args.bayesian:
        stat_dict = get_uncertainty_distribution(args, aggregation_type="separate")
    else:
        if args.separate_by_MS:
            merged_df = pd.read_csv(args.merged_file, sep="\t")
            merged_df=merged_df[["participant_id", "T1w_mri_field"]]


    results_dict = {}
    currentDirectory = pathlib.Path(args.model_path)
    currentPattern = "fold-*"

    for fold_dir in currentDirectory.glob(currentPattern):
        fold = int(str(fold_dir).split("-")[-1])
        cnn_classification_dir = os.path.join(args.model_path, 'fold-%i' % fold, 'cnn_classification')
        if args.selection_metrics is None:
            selection_metrics = []
            for f in os.scandir(cnn_classification_dir):
                metric=os.path.basename(os.path.normpath(f.path))
                if metric in possible_selection_metrics:
                    selection_metrics.append(metric)
        else:
            selection_metrics=args.selection_metrics

        for selection_metric in selection_metrics:
            if not selection_metric in results_dict.keys():
                results_dict[selection_metric] = pd.DataFrame()
            modes = ['train', 'validation']
            for ms_el in args.MS_list:
                modes.append('test_' + ms_el)

            for mode in modes:
                if "test" in mode:
                    if args.separate_by_MS and not args.bayesian:
                        test_diagnosis_path = os.path.join(cnn_classification_dir, selection_metric,
                                                           '%s_image_level_prediction.tsv' % (mode))
                        test_diagnosis_df = pd.read_csv(test_diagnosis_path, sep='\t')
                        prediction_column="predicted_label"

                        test_diagnosis_dict = get_test_prediction(test_diagnosis_df, merged_df)
                        for key in test_diagnosis_dict:
                            test_diagnosis_eval = evaluate_prediction(test_diagnosis_dict[key][["true_label"]].values.astype(int),
                                                                      test_diagnosis_dict[key][[prediction_column]].values.astype(int))
                            test_diagnosis_df = pd.DataFrame(test_diagnosis_eval, index=[0])
                            test_diagnosis_df = test_diagnosis_df.assign(fold=fold,
                                                                         mode=key)
                            results_dict[selection_metric] = pd.concat(
                                [results_dict[selection_metric], test_diagnosis_df],
                                axis=0)
                    else:
                        if args.bayesian:
                            for test_mode, values_df in stat_dict[fold][selection_metric].groupby("mode"):
                                if "test" in test_mode:

                                    prediction_column = "predicted_label_from_%s" % args.ba_inference_mode
                                    test_diagnosis_eval = evaluate_prediction(values_df[["true_label"]].values.astype(int),
                                                                              values_df[[prediction_column]].values.astype(int))
                                    test_diagnosis_df = pd.DataFrame(test_diagnosis_eval, index=[0])
                                    test_diagnosis_df = test_diagnosis_df.assign(fold=fold,
                                                                                 mode=test_mode)

                                    results_dict[selection_metric] = pd.concat(
                                                [results_dict[selection_metric], test_diagnosis_df],
                                                axis=0)

                else:
                    test_diagnosis_path = os.path.join(cnn_classification_dir, selection_metric,
                                                       '%s_image_level_metrics.tsv' % (mode))
                    if os.path.exists(test_diagnosis_path):
                        test_diagnosis_df = pd.read_csv(test_diagnosis_path, sep='\t')
                        test_diagnosis_df = test_diagnosis_df.assign(fold=fold,
                                                                 mode=mode)
                        test_diagnosis_df=test_diagnosis_df.drop(["total_loss", "image_id",], axis=1)

                        results_dict[selection_metric] = pd.concat([results_dict[selection_metric], test_diagnosis_df],
                                                           axis=0)

    resulting_metrics_dict = {}
    if aggregation_type=="average":
        for key in results_dict.keys():
            res_df = results_dict[key].drop(["fold"], axis=1)
            resulting_metrics_dict[key] = res_df.groupby(["mode"], as_index=False, sort=False).agg(np.mean)
        resulting_metrics_dict = {aggregation_type: resulting_metrics_dict}

    elif aggregation_type=="separate":
        for key in results_dict.keys():
            metric_dict = dict(list(results_dict[key].groupby("fold")))
            for fold in metric_dict.keys():
                if fold not in resulting_metrics_dict.keys():
                    resulting_metrics_dict[fold] = {}
                resulting_metrics_dict[fold][key] = metric_dict[fold]
    else:
        resulting_metrics_dict={"all":results_dict}
    return resulting_metrics_dict


def get_uncertainty_distribution(args, aggregation_type="average"):
    import pandas as pd
    import os
    import pathlib
    import numpy as np

    if args.separate_by_MS:
        merged_df = pd.read_csv(args.merged_file, sep="\t")
        merged_df=merged_df[["participant_id", "T1w_mri_field"]]
    currentDirectory = pathlib.Path(args.model_path)
    currentPattern = "fold-*"
    stat_dict = {}
    for fold_dir in currentDirectory.glob(currentPattern):
        fold = int(str(fold_dir).split("-")[-1])
        cnn_classification_dir = os.path.join(args.model_path, 'fold-%i' % fold, 'cnn_classification')

        if args.selection_metrics is None:
            selection_metrics = []
            for f in os.scandir(cnn_classification_dir):
                metric=os.path.basename(os.path.normpath(f.path))
                if metric in possible_selection_metrics:
                    selection_metrics.append(metric)
        else:
            selection_metrics = args.selection_metrics

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

                if "test" in mode and args.separate_by_MS:
                    test_diagnosis_dict = get_test_prediction(test_diagnosis_df, merged_df)
                    for key in test_diagnosis_dict:
                        test_diagnosis_dict[key] = test_diagnosis_dict[key].assign(fold=fold, mode=key)
                        stat_dict[selection_metric] = pd.concat([stat_dict[selection_metric], test_diagnosis_dict[key]],
                                                                axis=0)
                else:
                    test_diagnosis_df = test_diagnosis_df.assign(fold=fold, mode=mode)
                    stat_dict[selection_metric] = pd.concat([stat_dict[selection_metric], test_diagnosis_df], axis=0)
                # stat_dict[selection_metric].reset_index(inplace=True, drop=True)

    resulting_stat_dict = {}
    if aggregation_type=="average" or aggregation_type=="all":
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
        resulting_stat_dict = {aggregation_type: resulting_stat_dict}

    elif aggregation_type == "separate":
        for key in stat_dict.keys():
            metric_dict = dict(list(stat_dict[key].groupby("fold")))
            for fold in metric_dict.keys():
                if fold not in resulting_stat_dict.keys():
                    resulting_stat_dict[fold] = {}
                resulting_stat_dict[fold][key] = metric_dict[fold]

    return resulting_stat_dict


def get_history(args, aggregation_type="average"):
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
    if aggregation_type == "average":
        history_df = history_df[
            ["epoch", "balanced_accuracy_train", "loss_train", "balanced_accuracy_valid", "loss_valid"]]
        history_df = {aggregation_type: history_df.groupby("epoch", as_index=False).agg(np.mean)}

    elif aggregation_type == "all":
        history_df={aggregation_type:history_df}
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
        data[data_type] = eval("get_%s" % data_type)(args, args.aggregation_type)
    #data is now in format {data_type: {fold_0:, ...fold_n etc}}

    #toDo: turn off this function?
    if reshape_dict:
        # reshape data to format  {fold_0: {data_type_1:, ...data_type_i etc}}
        data = reshape_dictionary(data)
    return data
