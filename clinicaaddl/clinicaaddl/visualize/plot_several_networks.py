import matplotlib.pyplot as plt


def get_rows_and_cols(data):
    rows_matrix = {}
    cols_matrix = {}
    for data_type in data.keys():
        cols_matrix[data_type] = [selection_metric for selection_metric in data[data_type].keys()]

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

    for col, model in enumerate(list(data.keys())):
        ax = plt.subplot2grid(shape=figshape, loc=(row, col), fig=fig)
        plot_history_ax(ax, data[model], mode=args.history_mode, aggregation_type=args.aggregation_type)


def plot_results(args, data, fig, row, figshape):
    from .plot_utils import plot_results_ax, plot_results_agg_ax
    import seaborn as sns
    for col, model in enumerate(list(data.keys())):
        with sns.axes_style("whitegrid"):
            ax = plt.subplot2grid(shape=figshape, loc=(row, col), fig=fig)
            if args.aggregation_type is not "all":
                plot_results_ax(ax, data[model], args.result_metrics)
            else:
                plot_results_agg_ax(ax, data[model], args.result_metrics)
            # ax.set_title(str(chr(col+97)))




def plot_uncertainty_distribution(args, data, fig, row, figshape):
    from .plot_utils import plot_catplot_ax, set_ylims_axes
    axes = []

    for col, selection_mode in enumerate(list(data.keys())):
        for j, (mode, mode_group) in enumerate(data[selection_mode].groupby("mode", as_index=False, sort=False)):
            ax = plt.subplot2grid(shape=figshape, loc=(row+j, col), fig=fig)
            plot_catplot_ax(ax,mode_group, args.uncertainty_metric, args.ba_inference_mode, args.catplot_type)
            axes.append(ax)

    set_ylims_axes(axes)


def plot_combined_plots(args, saved_file_path, data=None):
    import matplotlib.pyplot as plt

    rows_matrix, cols_matrix, num_rows, num_cols = get_rows_and_cols(data)
    fig = plt.figure(figsize=((int(8 * num_cols), int(6 * num_rows))))

    row = 0
    for data_key in sorted(list(data.keys()), reverse=True):
        eval("plot_%s" % (data_key))(args, data=data[data_key], fig=fig, figshape=(num_rows, num_cols), row=row)
        row+=len(rows_matrix[data_key])

    # plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05, wspace=0.1, hspace=0.1)
    plt.subplots_adjust( left=0.05, right=0.95, top=0.95, bottom=0.05,hspace=0.3)
    # ax.set_title(str(chr(col + 97)))
    models_dict={}
    for i in range(num_cols):
        if args.hinder_titles:
            ax_title=str(chr(i + 97))
        else:
            model_name=cols_matrix[list(cols_matrix.keys())[0]][i]
            ax_title=model_name[len("subject_model-"):model_name.find("_preprocessing")]
        if "uncertainty_distribution" in cols_matrix.keys():
            fig.axes[i*2].set_title(ax_title)
        else:
            fig.axes[i].set_title(ax_title)

        models_dict[ax_title] = cols_matrix[list(cols_matrix.keys())[0]][i]
    if saved_file_path is not None:
        plt.savefig(saved_file_path)
    else:
        plt.show()

    plt.close()
    print(models_dict)


def plot_networks_generic(
        args,
        training_MS, models_list
):
    import pathlib
    import os
    import json
    import pandas as pd
    from .data_utils import get_data_generic
    data_list={}
    args.selection_metric=args.selection_metrics[0]
    args.history_mode=args.history_modes[0]
    for model_path in models_list:
        currentDirectory = pathlib.Path(model_path)
        path_params = os.path.join(currentDirectory, "commandline_train.json")

        with open(path_params, "r") as f:
            params = json.load(f)

        params['training MS'] = training_MS
        args.bayesian=params["bayesian"]
        model_name = os.path.basename(os.path.normpath(currentDirectory))
        args.model_path=model_path
        data = get_data_generic(args)
        for fold_key in data.keys():
            if not fold_key in data_list.keys():
                data_list[fold_key] = {}
            for datatype_key in data[fold_key].keys():
                if not datatype_key in data_list[fold_key].keys():
                    data_list[fold_key][datatype_key] = {}
                if datatype_key=="history":
                    data_list[fold_key][datatype_key][model_name] = data[fold_key][datatype_key]
                else:
                    data_list[fold_key][datatype_key][model_name]= data[fold_key][datatype_key][args.selection_metric]

            # data_list[model_name]=data[fold_key][args.selection_metrics[sel_key]]
    # if "results" or "uncertainty_distribution" in (args.data_types):
    #     folder_name = "%s-"%args.selection_metric
    # else:
    #     folder_name = "%s-"%args.history_mode
    folder_name=""

    for data_type in sorted(args.data_types):
        if data_type=="uncertainty_distribution":
            folder_name += '%s_uncertainty_%s' % (args.uncertainty_metric, args.catplot_type)
        else:
            folder_name += data_type
        folder_name += "_"


    if "results" or "uncertainty_distribution" in (args.data_types):
        model_name = args.selection_metric
    else:
        model_name = args.history_mode

    for fold_key in data_list.keys():

        if args.aggregation_type=="separate":
            folder_fold_name = os.path.join("separate_folds", "fold-%s"%fold_key)
        else:
            folder_fold_name = fold_key


        if args.output_path:
            saved_file_path = os.path.join(args.output_path, folder_fold_name, params["model"],  folder_name)
            os.makedirs(saved_file_path, exist_ok=True)
            saved_file_path=os.path.join(saved_file_path, model_name + '.png')
        else:
            saved_file_path=None

        plot_combined_plots(args, data=data_list[fold_key], saved_file_path=saved_file_path)
