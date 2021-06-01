class Plots():
    def __init__(self):
        self.description = "plots functionality"

    @staticmethod
    def barplots_loss(
            args, model_params, history, results,
            model_name, fold=None
    ):

        import os
        from pathlib import Path
        from .data_utils import get_results
        import pandas as pd
        import json
        from .plot_utils import barplots_with_loss

        folder_type = 'barplots_with_loss'

        # history = pd.read_csv(os.path.join(args.model_path, 'fold-%i' % fold, 'training.tsv'),
        #                       sep='\t')
        # results =get_results(args.model_path, args.MS_list, fold)
        path = os.path.join(args.output_path, folder_type)
        os.makedirs(path, exist_ok=True)
        file_name = model_name + '.png'
        barplots_with_loss(model_params, results, history, os.path.join(path, file_name))
        if args.save_best:
            best_model_filename = os.path.join(args.path_to_best, "best_model_results.json")
            if Path(best_model_filename).is_file():
                with open(best_model_filename, "r") as f:
                    reported_best_accuracies = json.load(f)
            else:
                reported_best_accuracies = {}
                for ms_el in args.MS_list:
                    reported_best_accuracies[ms_el] = {"max_value": 0}
            for ms_el in args.MS_list:
                for mode in results.keys():
                    if results[mode]["test_" + ms_el]["f1-score"][0] > \
                            reported_best_accuracies[ms_el]["max_value"]:
                        reported_best_accuracies[ms_el]["max_value"] = \
                            results[mode]["test_" + ms_el]["f1-score"][0]
                        reported_best_accuracies[ms_el]["prediction_path"] = args.model_path
                        reported_best_accuracies[ms_el]["params"] = model_params
                    reported_best_accuracies[ms_el]["model_name"] = model_name
            with open(best_model_filename, "w") as f:
                json.dump(reported_best_accuracies, f)

    @staticmethod
    def uncertainty_distribution(
            args, model_params, stat, results, model_name, fold=None,
    ):
        import os
        from .data_utils import get_baesian_stat, get_results

        from .plot_utils import plot_uncertainty_dist
        folder_type = '%s_uncertainty_histogram' % args.uncertainty_metric
        # stat = get_baesian_stat(args.model_path, args.MS_list, fold, args.uncertainty_metric)
        path = os.path.join(args.output_path, folder_type)
        os.makedirs(path, exist_ok=True)
        file_name = model_name + '.png'
        # results = get_results(args.model_path, args.MS_list, fold) if args.include_results else None
        plot_uncertainty_dist(model_params, stat, args.uncertainty_metric, separate_by_labels=args.separate_by_labels,
                              saved_file_path=os.path.join(path, file_name), results=results)

    @staticmethod
    def uncertainty_catplot(
            args, model_params, stat, results,
            model_name, fold=None,
    ):
        import os
        from .data_utils import get_baesian_stat, get_results

        from .plot_utils import plot_uncertainty_catplot
        folder_type = '%s_uncertainty_%s' % (args.uncertainty_metric, args.catplot_type)
        # stat = get_baesian_stat(args.model_path, args.MS_list, fold, args.uncertainty_metric)
        path = os.path.join(args.output_path, folder_type)
        os.makedirs(path, exist_ok=True)
        file_name = model_name + '.png'
        # results = get_results(args.model_path, args.MS_list, fold) if args.include_results else None
        plot_uncertainty_catplot(model_params, stat, args.uncertainty_metric, inference_mode=args.inference_mode,
                                 saved_file_path=os.path.join(path, file_name), results=results,
                                 catplot_type=args.catplot_type)


def plot_combined_plots(args, model_params, model_name, saved_file_path, data=None, ):
    import os


def create_data_dict(plot_types):
    data={}
    for key in plot_types.keys():
        data[key]={}
    return data


def plot_generic(
        args,
        magnet_strength,
):
    import pathlib
    import os
    import json
    import pandas as pd
    from .data_utils import get_baesian_stat, get_results

    currentDirectory = pathlib.Path(args.model_path)
    currentPattern = "fold-*"
    path_params = os.path.join(currentDirectory, "commandline_train.json")

    with open(path_params, "r") as f:
        params = json.load(f)

    params['training MS'] = magnet_strength
    model_name = os.path.basename(os.path.normpath(currentDirectory))



    folder_name = '-'.join(sorted(args.plot_types))
    if len(list(filter(lambda x: "uncertainty" in x, args.plot_types))) > 0:
        folder_name += '%s_uncertainty' % (args.uncertainty_metric)

    for fold_dir in currentDirectory.glob(currentPattern):

        fold = int(str(fold_dir).split("-")[-1])
        fold_name = 'fold-%i' % fold

        for plot_type in args.plot_types.keys():

            data[plot_type][fold_name] = pd.read_csv(os.path.join(args.model_path, fold_name, 'training.tsv'),
                                                  sep='\t')
            stat_list[fold_name] = get_baesian_stat(args.model_path, args.MS_list, fold, args.uncertainty_metric)

        results_list[fold_name] = get_results(args.model_path, args.MS_list, fold, stat_list)



        # plot results depending on
        if not args.average_fold:
            folder_fold_name = os.path.join("separate_folds", fold_name)
            path = os.path.join(args.output_path, folder_fold_name, folder_name)
            os.makedirs(path, exist_ok=True)

            plot_combined_plots(args, model_params=params, model_name=model_name, data=data,
                                saved_file_path=os.path.join(path, model_name + '.png'))

    if args.average_fold:
        pass
        # get_average





