from classify.bayesian_utils import bayesian_predictions
from visualize.plot import plot_generic
import os

if __name__ == "__main__":

    import pathlib
    import pandas as pd
    import os
    import json
    from visualize.plot import plot_generic

    folders = []
    # MS_main_list = ['1.5T', '3T', "1.5T-3T"]
    MS_main_list = ['1.5T']

    MS_list_dict = {'1.5T':['1.5T', '3T'], "3T": ['3T', '1.5T'], "1.5T-3T": ["1.5T-3T"]}
    # home_folder='/u/horlavanasta/MasterProject/'
    home_folder='/home/nastya/Documents/MasterProject/'

    isBayesian=True
    for MS in MS_main_list:
        print("____________________________________________________________________________________________")
        model_types = ["Conv5_FC3" ]
        MS_list = MS_list_dict[MS]

        results_folder_general =os.path.join(home_folder, 'Code/ClinicaTools/AD-DL/results/', "Experiments_Bayesian" if isBayesian else "Experiments", 'Experiments-' + MS)
        model_dir_general = os.path.join(home_folder,"DataAndExperiments/Experiments/Experiments-" + MS, "NNs_Bayesian" if isBayesian else "NNs")

        for network in model_types:
            model_dir = os.path.join(model_dir_general, network)
            # output_dir = pathlib.Path(output_dir)
            modelPatter = "subject_model*"
            folders = [f for f in pathlib.Path(model_dir).glob(modelPatter)]

            for f in folders[:]:
                results_dir=os.path.join(results_folder_general, network)
                # plot_generic(f, results_dir, MS, MS_list, "barplots_loss", **{"save_best":True, "path_to_best":results_dir})
                plot_generic(f, results_dir, MS, MS_list, "uncertainty_distribution", **{"separate_by_labels":True, "uncertainty_metric":"total_variance", "include_results":True})

                # for magnet_strength in MS_list:
                #     bayesian_predictions(predictions_path=f,
                #                                  prefix="test_"+magnet_strength,
                #                                  function="stat")

