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
    parser.add_argument("--average_fold", type=str2bool, default=True)
    parser.add_argument("--history_modes", nargs='+', default=["loss", "balanced_accuracy"])



    return parser.parse_args()



if __name__ == "__main__":

    import pathlib
    import pandas as pd
    import os
    import json
    from visualize.plot import plot_generic
    from visualize.data_utils import  get_data_generic

    folders = []
    # MS_main_list = ['1.5T', '3T', "1.5T-3T"]
    MS_main_list = ['3T']

    MS_list_dict = {'1.5T':['1.5T', '3T'], "3T": ['3T', '1.5T'], "1.5T-3T": ["1.5T-3T"]}
    # home_folder='/u/horlavanasta/MasterProject/'
    home_folder='/home/nastya/Documents/MasterProject/'
    data_types=["history", "uncertainty_distribution", "results"]
    data_types=["uncertainty_distribution", "results", "history"]

    isBayesian=True

    for MS in MS_main_list:
        print("____________________________________________________________________________________________")
        model_types = ["Conv5_FC3"]
        MS_list = MS_list_dict[MS]

        results_folder_general =os.path.join(home_folder, 'results/', "Experiments_Bayesian" if isBayesian else "Experiments", 'Experiments-' + MS)
        model_dir_general = os.path.join(home_folder,"DataAndExperiments/Experiments_3-fold/Experiments-" + MS, "NNs_Bayesian" if isBayesian else "NNs")

        for network in model_types:
            model_dir = os.path.join(model_dir_general, network)
            # output_dir = pathlib.Path(output_dir)
            modelPatter = "subject_model*"
            folders = [f for f in pathlib.Path(model_dir).glob(modelPatter)]

            for f in folders[:]:
                args=get_args(f, MS_list, data_types)
                args.get_test_from_bayesian=True
                args.ba_inference_mode="mode"
                args.MS_list=MS_list
                args.output_path=None
                args.result_metrics=["accuracy","sensitivity", "precision", "f1-score"]
                args.uncertainty_metric="total_variance"
                args.catplot_type = "violinplot"
                plot_generic(args, MS)

