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
    # parser.add_argument("--MS_list", nargs='+', default=MS_list)
    # parser.add_argument("--metrics",default=None)
    parser.add_argument("--aggregation_type", type=str, default="all")
    parser.add_argument("--history_modes", nargs='+', default=["loss", "balanced_accuracy"])



    return parser.parse_args()



if __name__ == "__main__":

    import pathlib
    import pandas as pd
    import os
    import json
    from visualize.plot import plot_generic
    from visualize.data_utils import get_data_generic
    from visualize.plot_several_networks import plot_networks_generic

    folders = []
    # MS_main_list = ['1.5T', '3T', "1.5T-3T"]
    MS_main_list = ["1.5T-3T"]

    MS_list_dict = {'1.5T':['1.5T', '3T'], "3T": ['3T', '1.5T'], "1.5T-3T": ["1.5T-3T"]}
    # home_folder='/u/horlavanasta/MasterProject/'
    home_folder='/home/nastya/Documents/MasterProject/'
    from classify.bayesian_utils import bayesian_predictions
    # data_types=["history", "uncertainty_distribution", "results"]
    data_types=["results", "history"]

    isBayesian=True

    for MS in MS_main_list:
        print("____________________________________________________________________________________________")
        model_types = ["ResNet18"]
        MS_list = MS_list_dict[MS]

        # results_folder_general =os.path.join(home_folder, 'results/', "Experiments_Bayesian" if isBayesian else "Experiments", 'Experiments-' + MS)
        model_dir_general = os.path.join(home_folder,"DataAndExperiments/Experiments_5-fold/Experiments-" + MS, "NNs_Bayesian" if isBayesian else "NNs")

        for network in model_types:
            model_dir = os.path.join(model_dir_general, network)
            # output_dir = pathlib.Path(output_dir)
            modelPatter = "subject_model*"
            folders = [f for f in pathlib.Path(model_dir).glob(modelPatter)]

            models_list=[]
            for f in folders[:]:
                args=get_args(f, MS_list, data_types)
                models_list.append(f)
                prefixes = ["test_" + magnet_strength for magnet_strength in MS_list]
                bayesian_predictions(model_path=f, prefixes=prefixes, function="stat")

                # plot_generic(args, MS)
                # data=get_data_generic(args, MS)
            args.ba_inference_mode = "mean"
            args.aggregation_type="all"
            args.MS_list = MS_list
            args.output_path = None
            args.bayesian = isBayesian
            args.merged_file = os.path.join(home_folder, "DataAndExperiments/Data/DataStat/merge.tsv")
            args.result_metrics = ["accuracy", "sensitivity", "precision", "f1-score"]
            args.uncertainty_metric = "total_variance"
            args.catplot_type = "violinplot"
            args.selection_metrics = None
            args.separate_by_MS = True
            args.selection_metrics="best_loss"
            plot_networks_generic(args, MS, models_list)