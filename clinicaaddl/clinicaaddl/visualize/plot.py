
class Plots():
    def __init__(self):
        self.description = "plots functionality"
    @staticmethod
    def barplots_loss(
            predictions_path,
            output_path,
            MS_list, model_params, fold,
            model_name,
            **kwargs
    ):
        # **kwargs store the following: save_best=True, path_to_best=None,


        import os
        from pathlib import Path
        from .data_utils import get_results
        import pandas as pd
        import json
        from .plot_utils import barplots_with_loss

        folder_type = 'barplots_with_loss'

        history = pd.read_csv(os.path.join(predictions_path, 'fold-%i' % fold, 'training.tsv'),
                              sep='\t')
        results =get_results(predictions_path, MS_list, fold)
        path = os.path.join(output_path, folder_type)
        os.makedirs(path, exist_ok=True)
        file_name = model_name + '.png'
        barplots_with_loss(model_params, results, history, os.path.join(path, file_name))
        if kwargs["save_best"]:
            best_model_filename =os.path.join(kwargs["path_to_best"], "best_model_results.json")
            if Path(best_model_filename).is_file():
                with open(best_model_filename, "r") as f:
                    reported_best_accuracies =json.load(f)
            else:
                reported_best_accuracies = {}
                for ms_el in MS_list:
                    reported_best_accuracies[ms_el] = {"max_value": 0}
            for ms_el in MS_list:
                for mode in results.keys():
                    if results[mode]["test_" + ms_el]["balanced_accuracy"][0] > \
                            reported_best_accuracies[ms_el]["max_value"]:
                        reported_best_accuracies[ms_el]["max_value"] = \
                            results[mode]["test_" + ms_el]["balanced_accuracy"][0]
                        reported_best_accuracies[ms_el]["prediction_path"] = str(predictions_path)
                        reported_best_accuracies[ms_el]["params"] = model_params
                    reported_best_accuracies[ms_el]["model_name"] = model_name
            with open(best_model_filename, "w") as f:
                json.dump(reported_best_accuracies, f)

    # @staticmethod
    # def variance_scatter(
    #         predictions_path,
    #         output_path,
    #         MS_list, model_params, fold,
    #         model_name,
    #         **kwargs
    # ):
    #     import os
    #     from .data_utils import get_baesian_stat
    #     folder_type = 'variance_scatter'
    #     from .plot_utils import scatter_variance_per_class
    #
    #     stat = get_baesian_stat(predictions_path, MS_list, fold, "class_variance")
    #     path = os.path.join(output_path, folder_type)
    #     os.makedirs(path, exist_ok=True)
    #     file_name = model_name + '.png'
    #     scatter_variance_per_class(model_params, stat,  os.path.join(path, file_name))
    #
    #     pass

    @staticmethod
    def uncertainty_distribution(
            predictions_path,
            output_path,
            MS_list, model_params, fold,
            model_name,
            **kwargs
    ):
        import os
        from .data_utils import get_baesian_stat, get_results

        from .plot_utils import plot_uncertainty_dist

        uncertainty_metric=kwargs["uncertainty_metric"]
        include_results=kwargs["include_results"]


        folder_type = '%s_uncertainty_histogram'%uncertainty_metric
        stat = get_baesian_stat(predictions_path, MS_list, fold, uncertainty_metric)
        path = os.path.join(output_path, folder_type)
        os.makedirs(path, exist_ok=True)
        file_name = model_name + '.png'
        results = get_results(predictions_path, MS_list, fold) if include_results else None
        plot_uncertainty_dist(model_params, stat, uncertainty_metric, separate_by_labels=kwargs["separate_by_labels"], saved_file_path=os.path.join(path, file_name), results=results)



def plot_generic(
        predictions_path,
        output_path,
        magnet_strength,
        MS_list=[],
        function=None,
        **kwargs
):
    import pathlib
    import os
    import json

    currentDirectory = pathlib.Path(predictions_path)
    currentPattern = "fold-*"
    path_params = os.path.join(currentDirectory, "commandline_train.json")

    with open(path_params, "r") as f:
        params = json.load(f)

    params['training MS'] = magnet_strength
    model_name = os.path.basename(os.path.normpath(currentDirectory))

    # loop depending the number of folds found in the model folder
    for fold_dir in currentDirectory.glob(currentPattern):
        fold = int(str(fold_dir).split("-")[-1])
        #plot results depending on
        getattr(Plots, function)(predictions_path=predictions_path, output_path=output_path,
              MS_list=MS_list, model_params=params, fold=fold, model_name=model_name, **kwargs)





