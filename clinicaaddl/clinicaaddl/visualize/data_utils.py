possible_selection_metrics = ["best_loss", "best_balanced_accuracy", "last_checkpoint"]

def get_results(args):
    import pandas as pd
    import os
    import pathlib

    columns = ["fold", "selection_metric", "mode"]
    if args.metrics is None:
        metrics=["accuracy", 'sensitivity', 'precision', 'f1-score']
    else:
        metrics=args.metrics

    for el in metrics:
        columns.append(el)

    # if args.get_test_from_bayesian:
    #     stat_df=get_uncertainty_distribution(args)

    # results_df= pd.DataFrame(columns = columns )
    results_df= pd.DataFrame( )

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
            modes = ['train', 'validation']
            for ms_el in args.MS_list:
                modes.append('test_' + ms_el)

            for mode in modes:
                if "test" in mode and args.get_test_from_bayesian:

                    #toDo: read from bayes stat
                    test_diagnosis_path = os.path.join(cnn_classification_dir, selection_metric,
                                                       '%s_image_level_metrics.tsv' % (mode))
                    test_diagnosis_df = pd.read_csv(test_diagnosis_path, sep='\t')
                    test_diagnosis_df = test_diagnosis_df.assign(selection_metric=selection_metric, mode=mode)

                else:
                    test_diagnosis_path = os.path.join(cnn_classification_dir, selection_metric,
                                                       '%s_image_level_metrics.tsv' % (mode))
                    test_diagnosis_df = pd.read_csv(test_diagnosis_path, sep='\t')
                    test_diagnosis_df = test_diagnosis_df.assign(fold=fold, selection_metric=selection_metric, mode=mode)
                results_df = pd.concat([results_df, test_diagnosis_df[metrics]], axis=0)

    return results_df

def get_uncertainty_distribution(args):
    import pandas as pd
    import os
    import pathlib


    selection_metrics = []
    currentDirectory = pathlib.Path(args.model_path)
    currentPattern = "fold-*"
    # stat_df = pd.DataFrame( columns = ["fold", "selection_metric", "prefix",  "participant_id", "session_id", "true_label", "predicted_label_from_mean", "predicted_label_from_mode", "class_variance", "total_variance", "entropy", "NLL"])
    stat_df = pd.DataFrame()

    for fold_dir in currentDirectory.glob(currentPattern):
        fold = int(str(fold_dir).split("-")[-1])

        cnn_classification_dir = os.path.join(args.model_path, 'fold-%i' % fold, 'cnn_classification')
        for f in os.scandir(cnn_classification_dir):
            if os.path.basename(os.path.normpath(f.path)) in possible_selection_metrics:
                selection_metrics.append(os.path.basename(os.path.normpath(f.path)))

        for selection_metric in selection_metrics:
            modes = ['test_' + ms_el for ms_el in args.MS_list]

            for mode in modes:
                test_diagnosis_path = os.path.join(cnn_classification_dir, selection_metric,
                                                   "bayesian_statistics", '%s_image_level_stats.tsv' % (mode))
                test_diagnosis_df = pd.read_csv(test_diagnosis_path, sep='\t')
                test_diagnosis_df["class_variance"] = test_diagnosis_df["class_variance"].apply(lambda x: x[1:-1].split()).apply(lambda x: [float(i) for i in x])
                test_diagnosis_df=test_diagnosis_df.assign(fold=fold, selection_metric=selection_metric, mode=mode)
                stat_df=pd.concat([stat_df, test_diagnosis_df], axis=0 )

    return stat_df

def get_history(args):
    import pandas as pd
    import os
    import pathlib
    currentDirectory = pathlib.Path(args.model_path)
    currentPattern = "fold-*"
    history_df=pd.DataFrame()
    for fold_dir in currentDirectory.glob(currentPattern):
        fold = int(str(fold_dir).split("-")[-1])
        history= pd.read_csv(os.path.join(args.model_path, 'fold-%i' % fold, 'training.tsv'), sep='\t')
        history = history.assign(fold=fold)
        history_df = pd.concat([history_df, history], axis=0)

    return history_df


def get_data_generic(args, average_fold=True):

    data = {}
    for data_type in args.data_types:
        data[data_type] = eval("get_%s"%data_type)(args)






