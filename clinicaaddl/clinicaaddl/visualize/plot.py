import matplotlib.pyplot as plt


class Plots():
    def __init__(self):
        self.description = "plots functionality"

    # @staticmethod
    # def barplots_loss(
    #         args, model_params, history, results,
    #         model_name, fold=None
    # ):

        # import os
        # from pathlib import Path
        # from .data_utils import get_results
        # import pandas as pd
        # import json
        # from .plot_utils import barplots_with_loss
        #
        # folder_type = 'barplots_with_loss'
        #
        # # history = pd.read_csv(os.path.join(args.model_path, 'fold-%i' % fold, 'training.tsv'),
        # #                       sep='\t')
        # # results =get_results(args.model_path, args.MS_list, fold)
        # path = os.path.join(args.output_path, folder_type)
        # os.makedirs(path, exist_ok=True)
        # file_name = model_name + '.png'
        # barplots_with_loss(model_params, results, history, os.path.join(path, file_name))
        # if args.save_best:
        #     best_model_filename = os.path.join(args.path_to_best, "best_model_results.json")
        #     if Path(best_model_filename).is_file():
        #         with open(best_model_filename, "r") as f:
        #             reported_best_accuracies = json.load(f)
        #     else:
        #         reported_best_accuracies = {}
        #         for ms_el in args.MS_list:
        #             reported_best_accuracies[ms_el] = {"max_value": 0}
        #     for ms_el in args.MS_list:
        #         for mode in results.keys():
        #             if results[mode]["test_" + ms_el]["f1-score"][0] > \
        #                     reported_best_accuracies[ms_el]["max_value"]:
        #                 reported_best_accuracies[ms_el]["max_value"] = \
        #                     results[mode]["test_" + ms_el]["f1-score"][0]
        #                 reported_best_accuracies[ms_el]["prediction_path"] = args.model_path
        #                 reported_best_accuracies[ms_el]["params"] = model_params
        #             reported_best_accuracies[ms_el]["model_name"] = model_name
        #     with open(best_model_filename, "w") as f:
        #         json.dump(reported_best_accuracies, f)

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




def get_rows_and_cols(data):
    rows_matrix = {}
    cols_matrix = {}
    for data_type in data.keys():
        cols_matrix[data_type] = [selection_metric.replace("_", " ") for selection_metric in data[data_type].keys()]
        if data_type == "history":
            cols_matrix[data_type] = ["loss", "balanced_accuracy"]
        else:
            cols_matrix[data_type] = [selection_metric.replace("_", " ") for selection_metric in data[data_type].keys()]

        if data_type == "uncertainty_distribution":
            rows_matrix[data_type] = [test_MS.replace("_", " ") for test_MS in
                                      list(data[data_type][list(data[data_type].keys())[0]].groupby("mode", as_index=False, sort=False).groups.keys())]
        else:
            rows_matrix[data_type] = [None]

    num_rows = sum([len(rows_matrix[row]) for row in rows_matrix.keys()])
    num_cols = max([len(cols_matrix[col]) for col in cols_matrix.keys()])
    return rows_matrix, cols_matrix, num_rows, num_cols



def plot_history(args, data, fig, row, figshape):
    from .plot_utils import plot_history_ax

    for col, history_mode in enumerate(args.history_modes):
        ax = plt.subplot2grid(shape=figshape, loc=(row, col), fig=fig)
        plot_history_ax(ax, data, mode=history_mode)
    return row + 1


def plot_results(args, data, fig, row, figshape):
    from .plot_utils import plot_results_ax

    for col, selection_mode in enumerate(list(data.keys())):
        ax = plt.subplot2grid(shape=figshape, loc=(row, col), fig=fig)
        plot_results_ax(ax, data[selection_mode], args.result_metrics)
        ax.set_title(selection_mode)
    return row + 1


def plot_uncertainty_distribution(args, data, fig, row, figshape):
    from .plot_utils import plot_catplot_ax, set_ylims_axes
    axes = []

    for col, selection_mode in enumerate(list(data.keys())):
        for j, (mode, mode_group) in enumerate(data[selection_mode].groupby("mode", as_index=False, sort=False)):
            ax = plt.subplot2grid(shape=figshape, loc=(row+j, col), fig=fig)
            plot_catplot_ax(ax,mode_group, args.uncertainty_metric, args.ba_inference_mode, args.catplot_type )
            ax.set_title(selection_mode+"; "+mode)

            axes.append(ax)


    set_ylims_axes(axes)
    return row+2

def plot_combined_plots(args, model_params, saved_file_path, data=None):
    import matplotlib.pyplot as plt

    readable_params = ['model', 'data_augmentation', 'batch_size', 'learning_rate', "loss", 'training MS']

    rows_matrix, cols_matrix, num_rows, num_cols = get_rows_and_cols(data)
    fig = plt.figure(figsize=((int(8 * num_cols), int(6 * num_rows))))

    row = 0
    for data_key in data.keys():
        row = eval("plot_%s" % (data_key))(args, data=data[data_key], fig=fig, figshape=(num_rows, num_cols), row=row)


    str_suptitle = "\n Params: "
    for i, line in enumerate(readable_params):
        str_suptitle += line + ': ' + str(model_params[line]) + "; "
    str_suptitle +="\n"

    plt.suptitle(str_suptitle)

    # plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05, wspace=0.1, hspace=0.1)
    plt.subplots_adjust( left=0.05, right=0.95, top=0.95, bottom=0.05,)

    if saved_file_path is not None:
        plt.savefig(saved_file_path)
    else:
        plt.show()

    plt.close()


def plot_generic(
        args,
        training_MS,
):
    import pathlib
    import os
    import json
    import pandas as pd
    from .data_utils import get_data_generic

    currentDirectory = pathlib.Path(args.model_path)
    path_params = os.path.join(currentDirectory, "commandline_train.json")

    with open(path_params, "r") as f:
        params = json.load(f)

    params['training MS'] = training_MS
    args.bayesian=params["bayesian"]
    model_name = os.path.basename(os.path.normpath(currentDirectory))

    folder_name = ''
    for data_type in sorted(args.data_types):
        if data_type=="uncertainty_distribution":
            folder_name += '%s_uncertainty' % (args.uncertainty_metric)
        else:
            folder_name += data_type

    data = get_data_generic(args)

    for fold_key in data.keys():

        if not args.average_fold:
            folder_fold_name = os.path.join("separate_folds", fold_key)
        else:
            folder_fold_name = fold_key

        if args.output_path:
            saved_file_path = os.path.join(args.output_path, folder_fold_name, folder_name)
            os.makedirs(saved_file_path, exist_ok=True)
            saved_file_path=os.path.join(saved_file_path, model_name + '.png')
        else:
            saved_file_path=None

        plot_combined_plots(args, model_params=params, data=data[fold_key], saved_file_path=saved_file_path)
