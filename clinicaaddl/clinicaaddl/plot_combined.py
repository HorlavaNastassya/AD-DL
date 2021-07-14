from classify.bayesian_utils import bayesian_predictions
from visualize.plot import plot_generic
import os
from cli import str2bool


def get_args():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_types", nargs='+', default=["results", "history", "uncertainty_distribution"])
    parser.add_argument("--aggregation_type", type=str, default="all")
    parser.add_argument("--history_mode", default="loss")
    parser.add_argument("--selection_metric",  default="best_loss")

    parser.add_argument(
        '--hinder_titles',
        help='''indicates whether to substitute long name with a, b, ....''', type=str2bool,
        default=True)
    
    parser.add_argument("--preprocessing",  default="linear", choices=["linear", "none"], type=str)
    parser.add_argument(
        '--bayesian', type=str2bool,
        default=True)

    return parser.parse_args()

def set_args(args, MS, MS_list, results_dir, merged_file,  inference_mode=None):
    
    args.ba_inference_mode = inference_mode
    args.aggregation_type="all"
    args.MS_list = MS_list
    args.output_path = results_dir
    args.merged_file = merged_file
    args.result_metrics = ["sensitivity", "precision", "accuracy", "f1-score"]
    args.uncertainty_metric = "total_variance"
    args.catplot_type = "violinplot"
    args.hinder_titles=True
    if MS=="1.5T-3T":
        args.separate_by_MS=True
    else:
        args.separate_by_MS=False
    return args
    

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

#     MS_main_list = ['1.5T', "1.5T-3T", '3T']
#     MS_main_list = ['1.5T', "1.5T-3T"]
    MS_main_list = ["3T"]


    num_folds=5
    MS_list_dict = {'1.5T':['1.5T', '3T'], "3T": ['3T', '1.5T'], "1.5T-3T": ["1.5T-3T"]}
    home_folder='/u/horlavanasta/MasterProject/'    
    merged_file=os.path.join(home_folder,"DataAndExperiments/Data/DataStat", "merge.tsv")
    inference_modes=["mean"]
    args=get_args()
    
    
    for MS in MS_main_list[:]:
        print("MS %s \n ____________________________________________________________________________________________"%MS)
        model_types = [ "ResNet18", "SEResNet18", "ResNet18Expanded", "SEResNet18Expanded", "Conv5_FC3", "ResNet50", "SEResNet50"]
        MS_list = MS_list_dict[MS]
        
        results_folder_general =os.path.join(home_folder, 'Code/ClinicaTools/AD-DL/results_combined/', "Experiments_%d-fold"%num_folds, "Experiments_Bayesian" if args.bayesian else "Experiments", 'Experiments-' + MS)
        
        model_dir_general = os.path.join(home_folder,"DataAndExperiments/Experiments_%d-fold/Experiments-%s"%(num_folds, MS), "NNs_Bayesian" if args.bayesian else "NNs")
#         print(args.preprocessing)

        for network in model_types[:]:
            
            model_dir = os.path.join(model_dir_general, network)
            modelPatter = "subject_model*"
            folders = [f for f in pathlib.Path(model_dir).glob(modelPatter)]
            
            for augm in ["True", "False"]:
                models_list=[]
                results_dir=os.path.join(results_folder_general, "augm%s"%augm)
                for f in folders[:]:
                    if check_complete_test(f, num_folds=num_folds, MS_list=MS_list) and "_preprocessing-%s"%args.preprocessing in str(f) and "_augm%s"%augm in str(f):
                        if args.bayesian and not check_baesian_stat(f, num_folds=num_folds, MS_list=MS_list):
                            prefixes = ["test_" + magnet_strength for magnet_strength in MS_list]
                            bayesian_predictions(model_path=f, prefixes=prefixes, function="stat") 
                        models_list.append(f)
                        
                if args.bayesian: 
                    if "uncertainty_distribution" in args.data_types:
                        args.data_types.remove("uncertainty_distribution")
                        
                    for inference_mode in inference_modes:
                        results_dir=os.path.join(results_dir, "%s_inference"%inference_mode)
                        print(results_dir)
                        args=set_args(args,MS, MS_list, results_dir, merged_file,inference_mode)
                        plot_networks_generic(args, MS, models_list)
                else:
                    if "uncertainty_distribution" in args.data_types:
                        args.data_types.remove("uncertainty_distribution")
                    print(results_dir)
                    args=set_args(args, MS, MS_list, results_dir, merged_file,  inference_mode=None)
                    plot_networks_generic(args, MS, models_list)
                