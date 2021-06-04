import matplotlib.pyplot as plt


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



def plot_results(args, data, fig, row, figshape):
    from .plot_utils import plot_results_ax

    for col, selection_mode in enumerate(list(data.keys())):
        ax = plt.subplot2grid(shape=figshape, loc=(row, col), fig=fig)
        plot_results_ax(ax, data[selection_mode], args.result_metrics)
        ax.set_title(selection_mode)



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


def plot_combined_plots(args, model_params, saved_file_path, data=None):
    import matplotlib.pyplot as plt

    readable_params = ['model', 'data_augmentation', 'batch_size', 'learning_rate', "loss", 'training MS']

    rows_matrix, cols_matrix, num_rows, num_cols = get_rows_and_cols(data)
    fig = plt.figure(figsize=((int(8 * num_cols), int(6 * num_rows))))

    row = 0
    for data_key in sorted(list(data.keys()), reverse=True):
        eval("plot_%s" % (data_key))(args, data=data[data_key], fig=fig, figshape=(num_rows, num_cols), row=row)
        row+=len(rows_matrix[data_key])

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
            folder_name += '%s_uncertainty_%s' % (args.uncertainty_metric, args.catplot_type)
        else:
            folder_name += data_type
        folder_name += "_"

    data = get_data_generic(args)

    for fold_key in data.keys():

        if not args.average_fold:
            folder_fold_name = os.path.join("separate_folds", "fold-%s"%fold_key)
        else:
            folder_fold_name = fold_key

        if args.output_path:
            saved_file_path = os.path.join(args.output_path, folder_fold_name, folder_name)
            os.makedirs(saved_file_path, exist_ok=True)
            saved_file_path=os.path.join(saved_file_path, model_name + '.png')
        else:
            saved_file_path=None

        plot_combined_plots(args, model_params=params, data=data[fold_key], saved_file_path=saved_file_path)
