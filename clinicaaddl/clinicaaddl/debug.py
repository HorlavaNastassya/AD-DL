from classify.bayesian_utils import bayesian_predictions
from visualize.plot import plot_generic
import os
from cli import str2bool
def get_args(model_path, MS_list, data_types):
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", default=model_path)
    # parser.add_argument("data_types", nargs='+', default=["results", "history", "uncertainty_distribution"])
    parser.add_argument("--data_types", nargs='+', default=data_types)
    parser.add_argument("--aggregation_type", type=str, default="all")
    parser.add_argument("--history_modes", nargs='+', default=["loss", "balanced_accuracy"])
    parser.add_argument(
        '--hinder_titles',
        help='''indicates whether to substitute long name with a, b, ....''', type=str2bool,
        default=True)
    return parser.parse_args()


def check_history(model_path, num_folds):
    from visualize.data_utils import get_data_generic
    return os.path.exists(os.path.join(model_path, "status.txt"))
    
def check_results(model_path, MS_list, num_folds):
    import os
    import pathlib
    import numpy as np
    currentDirectory = pathlib.Path(model_path)
    currentPattern = "fold-*"
    flag=True
    for fold_dir in currentDirectory.glob(currentPattern):
        fold = int(str(fold_dir).split("-")[-1])

        selection_metrics = ["best_loss", "best_balanced_accuracy", "last_checkpoint"]
        cnn_classification_dir = os.path.join(model_path, 'fold-%i' % fold, 'cnn_classification')
        
        for selection_metric in selection_metrics:
            modes = ['train', 'validation']
            for ms_el in MS_list:
                modes.append('test_' + ms_el)
                
            for mode in modes:
                if not os.path.exists(os.path.join(cnn_classification_dir, selection_metric,
                                                       '%s_image_level_metrics.tsv' % (mode))):
                    flag=False
                
    return flag
    
def check_complete_test(model_path, num_folds, MS_list):
    import json
    path_params = os.path.join(model_path, "commandline_train.json")
    return (check_history(model_path, num_folds) and check_results(model_path, MS_list, num_folds))
    
def check_baesian_stat(model_path, MS_list, num_folds):
    import os
    import pathlib
    import numpy as np
    currentDirectory = pathlib.Path(model_path)
    currentPattern = "fold-*"
    flag=True
    for fold_dir in currentDirectory.glob(currentPattern):
        fold = int(str(fold_dir).split("-")[-1])

        selection_metrics = ["best_loss", "best_balanced_accuracy", "last_checkpoint"]
        cnn_classification_dir = os.path.join(model_path, 'fold-%i' % fold, 'cnn_classification')
        
        for selection_metric in selection_metrics:
            modes = ['test_' + ms_el for ms_el in MS_list]
                
            for mode in modes:
                if not os.path.exists(os.path.join(cnn_classification_dir, selection_metric,
                                                       '%s_image_level_stats.tsv' % (mode))):
                    flag=False
                
    return flag

if __name__ == "__main__":

    import pathlib
    import pandas as pd
    import os
    import json
    from visualize.plot import plot_generic
    from visualize.data_utils import get_data_generic
    from visualize.plot_several_networks import plot_networks_generic
    from classify.bayesian_utils import bayesian_predictions
    from copy import deepcopy
    folders = []
    MS_main_list = ["1.5T-3T"]

    MS_list_dict = {'1.5T':['1.5T', '3T'], "3T": ['3T', '1.5T'], "1.5T-3T": ["1.5T-3T"]}
    home_folder='/home/nastya/Documents/MasterProject/'
    data_types=["uncertainty_distribution"]
    # data_types=["results", "history"]

    isBayesian=True

    for MS in MS_main_list:
        print("____________________________________________________________________________________________")
        model_types = ["ResNet18"]
        MS_list = MS_list_dict[MS]

        # results_folder_general =os.path.join(home_folder, 'results/', "Experiments_Bayesian" if isBayesian else "Experiments", 'Experiments-' + MS)
        model_dir_general = os.path.join(home_folder,"DataAndExperiments/Experiments_5-fold/Experiments-" + MS, "NNs_Bayesian" if isBayesian else "NNs")

        for network in model_types:
            model_dir = os.path.join(model_dir_general, network)
            print(network)
            print("______________________________________________________________________________-")
            # output_dir = pathlib.Path(output_dir)
            modelPatter = "subject_model*"
            folders = [f for f in pathlib.Path(model_dir).glob(modelPatter)]

            models_list=''
            for f in folders[:1]:
                args=get_args(f, MS_list, data_types)
                models_list+="%s;"%f
                # prefixes = ["test_" + magnet_strength for magnet_strength in MS_list]
                # bayesian_predictions(model_path=f, prefixes=prefixes, function="stat")
                # plot_generic(args, MS)
                # data=get_data_generic(args, MS)

                args.ba_inference_mode = "mean"
                args.aggregation_type="all"
                args.MS_list = MS_list
                args.catplot_type = "violinplot"

                args.output_path = None
                args.bayesian = isBayesian
                args.merged_file = os.path.join(home_folder, "DataAndExperiments/Data/DataStat/merge.tsv")
                args.result_metrics = ["sensitivity", "precision", "accuracy", "f1-score"]
                columns = deepcopy(args.result_metrics)
                columns.append("mode")

                args.uncertainty_metric = "total_variance"

                args.separate_by_MS = True
                args.selection_metrics=["best_balanced_accuracy", "best_loss", "last_checkpoint"]
                args.selection_metrics=["best_balanced_accuracy"]


                MS_list_printed=MS_list if not args.separate_by_MS else ["1.5T", "3T"]
                printed_str=''
                plot_generic(args, MS)

                # for selection_metric in args.selection_metrics:
                #     print(selection_metric)
                #     for MS_el in MS_list_printed:
                #
                #         tmp2 = data[selection_metric].loc[data[selection_metric]["mode"] == "test_%s"%MS_el][args.result_metrics]
                #         printed_str=MS_el+" & "
                #         for res_metr in args.result_metrics:
                #             printed_str=printed_str+str(tmp2[res_metr].values[0]*100)+' & '
                #         print(printed_str)







