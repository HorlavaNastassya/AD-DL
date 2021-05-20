from classify.bayesian_utils import bayesian_predictions
from visualize.plot import plot_generic
import os
from visualize.plot import  plot_generic

if __name__ == "__main__":

    import pathlib
    import pandas as pd
    import os
    import json

    progress_list = []
    folders = []
    MS_main_list = ['1.5T']

    isBayesian=True
    for MS in MS_main_list:
        print("____________________________________________________________________________________________")
        model_types = ["Conv5_FC3"]
        MS_list = ['1.5T', '3T'] if MS == '1.5T' else ['3T', '1.5T']

        results_folder_general = '/home/nastya/Documents/MasterProject/results/Experiments/Experiments-' + MS
        model_dir_general = os.path.join("/home/nastya/Documents/MasterProject/Experiments/Experiments-" + MS, "NNs_Bayesian" if isBayesian else "NNs")

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
