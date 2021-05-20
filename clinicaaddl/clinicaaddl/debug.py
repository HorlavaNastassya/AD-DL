from classify.bayesian_utils import bayesian_predictions
from visualize.plot import plot_generic
import os

if __name__ == "__main__":

    import pathlib
    import pandas as pd
    import os
    import json
    from clinicaaddl.visualize.plot import plot_generic

    folders = []
    MS_main_list = ['1.5T', '3T', "1.5T-3T"]
    MS_list_dict = {'1.5T':['1.5T', '3T'], "3T": ['3T', '1.5T'], "1.5T-3T": ["1.5T-3T"]}
    home_folder='/u/horlavanasta/MasterProject/'

    isBayesian=True
    for MS in MS_main_list:
        print("____________________________________________________________________________________________")
        model_types = [ "ResNet18", "SEResNet18", "ResNet18Expanded", "SEResNet18Expanded", "Conv5_FC3" ]
        MS_list = MS_list_dict[MS]

        results_folder_general =os.path.join(home_folder, 'Code/ClinicaTools/AD-DL/results/', "Experiments_Bayesian" if isBayesian else "Experiments", 'Experiments-' + MS)
        model_dir_general = os.path.join(home_folder,"DataAndExperiments/Experiments/Experiments-" + MS, "NNs_Bayesian" if isBayesian else "NNs")

        for network in model_types:
            model_dir = os.path.join(model_dir_general, network)
            # output_dir = pathlib.Path(output_dir)
            modelPatter = "subject_model*"
            folders = [f for f in pathlib.Path(model_dir).glob(modelPatter)]

            for f in folders[:]:
                results_dir=os.path.join(results_folder_general,network)
                plot_generic(f, results_dir, MS, MS_list, "barplots_loss", **{"save_best":True, "path_to_best":results_dir})


        # for network_type in model_types:
        #     for ms_el in MS_list:
        #
        #         #     print(reported_best_accuracies[network_type]["params"])
        #         if reported_best_accuracies[network_type][ms_el]["max_value"] > 0.0:
        #             plot_loss_with_results(reported_best_accuracies[network_type][ms_el]["params"],
        #                                    reported_best_accuracies[network_type][ms_el]["results"],
        #                                    reported_best_accuracies[network_type][ms_el]["history"])

    # for predictions_path in predictions_path_list:
    #     bayesian_predictions(predictions_path=predictions_path,
    #                              prefix="test_1.5T",
    #                              selection_metrics=["loss", "balanced_accuracy", "last_checkpoint"],
    #                              function="stat")

    # func=getattr(
