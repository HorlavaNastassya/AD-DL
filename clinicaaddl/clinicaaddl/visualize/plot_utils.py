import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import json
import numpy as np
from matplotlib import cm


def plot_history_ax(ax, history, mode):
    import seaborn as sns
    def find_best(arr, ismin=True):
        arr = np.array(arr)
        if ismin:
            best_loss_idx_train = np.where(arr == np.amin(arr))[0][0]
        else:
            best_loss_idx_train = np.where(arr == np.amax(arr))[0][0]
        return best_loss_idx_train, arr[best_loss_idx_train]

    sns.lineplot(data=history, ax=ax, x="epoch", y=mode + "_train", label='train ' + mode, legend="brief")
    sns.lineplot(data=history, ax=ax, x="epoch", y=mode + "_valid", label='validation ' + mode, legend="brief")

    idx, val = find_best(history[mode + "_valid"], mode == 'loss')
    ax.plot(idx, val, 'o', color='black')

    if mode == 'loss':
        ax.set_ylim(bottom=-0.001, top=0.5)
    if mode == 'balanced_accuracy':
        ax.set_ylim(bottom=-0.001, top=1.1)
    ax.set_title(mode)

def plot_results_ax(ax, results, columns):

    def reshape_results(results, columns):
        reshaped_columns = ["mode", "metric", "value"]
        reshaped_df = pd.DataFrame(columns=reshaped_columns)
        for col in list(results[columns].columns):
            for idx, row in results.iterrows():
                new_row = [[row["mode"], col, row[col]]]
                row_df = pd.DataFrame(new_row, columns=reshaped_columns)
                reshaped_df = pd.concat([reshaped_df, row_df], axis=0)

        return reshaped_df

    def show_values_on_bars(axs):
        def _show_on_single_plot(ax, factor=0.025):
            for p in ax.patches:
                _x = p.get_x() + p.get_width() / 2
                _y = p.get_y() + p.get_height() +ax.get_ylim()[-1]*factor
                value = '{:.2f}'.format(p.get_height())
                ax.text(_x, _y, value, ha="center")

        if isinstance(axs, np.ndarray):
            for idx, ax in np.ndenumerate(axs):
                _show_on_single_plot(ax)
        else:
            _show_on_single_plot(axs)

    import seaborn as sns
    import pandas as pd

    results=reshape_results(results, columns)

    # sns.barplot(data=results,hue="metric", x="mode", y="value", ax=ax, palette=sns.color_palette("Paired"), linewidth=1.5)
    sns.barplot(data=results,hue="metric", x="mode", y="value", ax=ax, palette="Paired", linewidth=1.5)

    ax.set_ylim(bottom=-0.001, top=1.1)
    show_values_on_bars(ax)


def plot_catplot_ax(ax, data, uncertainty_metric, inference_mode, catplot_type):
    import seaborn as sns

    prediction_column = "predicted_label_from_%s" %inference_mode
    data["Prediction is correct"] = data.apply(
        lambda row: row["true_label"] == row[prediction_column], axis=1)
    arguments = {"data": data, "x": "true_label", "y": uncertainty_metric,
                 "hue": "Prediction is correct", "palette": "Set2", "ax": ax,
                 "hue_order": [True, False]}

    if catplot_type == "violinplot":
        arguments["split"] = True
        arguments["scale"] = "count"

    if catplot_type == "stripplot":
        arguments["dodge"] = True
        arguments["size"] = 4
        arguments["linewidth"] = 1
    getattr(sns, catplot_type)(**arguments)

def annotate(axes, cols, rows):
    for ax, col in zip(axes[0], cols):
        ax.annotate(col, xy=(0.5, 1), xytext=(0, 5),
                    xycoords='axes fraction', textcoords='offset points',
                    fontsize=20, ha='center', va='baseline')

    for ax, row in zip(axes[:, 0], rows):
        ax.annotate(row, xy=(0, 0.5), xytext=(-ax.yaxis.labelpad - 5, 0),
                    xycoords=ax.yaxis.label, textcoords='offset points',
                    fontsize=15, ha='right', va='center')

def set_ylims_axes(axes):
    min_ylim=min(ax.get_ylim()[0] for ax in axes)
    max_ylim=max(ax.get_ylim()[1] for ax in axes)
    for ax in axes:
        ax.set_ylim(bottom=min_ylim, top=max_ylim)


def plot_hist(axes, stat, uncertainty_metric, rows, cols, separate_by_labels):
    import seaborn as sns
    import numpy as np

    xlim_list={"left_limit":[], "right_limit":[]}

    for i, selection_metric in enumerate(stat.keys()):
        for j, test_MS in enumerate(stat[selection_metric].keys()):
            st = stat[selection_metric][test_MS]
            sns.histplot(data=st, x=st[uncertainty_metric], hue=st.true_label.values if separate_by_labels else None,
                         ax=axes[j][i], stat="probability", bins=10)
            xlim_list["left_limit"].append(min(axes[j][i].get_xlim()))
            xlim_list["right_limit"].append(max(axes[j][i].get_xlim()))

    #set_xlim for all histogram plots
    xlim_left,xlim_right =np.min(xlim_list["left_limit"]), np.max(xlim_list["right_limit"])
    for i, row in enumerate(rows):
        for j, col in enumerate(cols):
            axes[i][j].set_xlim(left=xlim_left, right=xlim_right)

    annotate(axes, cols, rows)


def plot_uncertainty_dist(model_params, stat,  uncertainty_metric, separate_by_labels=False, saved_file_path=None, results=None):
    import matplotlib

    font = {
        # 'family': 'normal',
            'weight': 'bold',
            'size': 24}

    matplotlib.rc('font', **font)

    cols = [selection_metric.replace("_", " ") for selection_metric in stat.keys()]
    rows = [test_MS.replace("_", " ") for test_MS in stat[list((stat.keys()))[0]].keys()]

    str_suptitle = "Params: "
    for i, line in enumerate(readable_params):
        str_suptitle += line + ': ' + str(model_params[line]) + "; "

    num_rows=len(rows)+1 if results else len(rows)
    fig, axes = plt.subplots(num_rows, len(cols), figsize=(int(20 * len(cols)), int(16 * num_rows)), sharey="row")

    plot_hist(axes, stat,uncertainty_metric, rows, cols, separate_by_labels)

    if results:
        for k, mode in enumerate(results.keys()):
            plot_bar_plots(axes[-1][k], results[mode], mode)

    plt.suptitle(str_suptitle, fontsize="large", fontweight="bold")
    plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05, wspace=0.1, hspace=0.1)
    if saved_file_path is not None:
        plt.savefig(saved_file_path)
    else:
        plt.show()
    plt.close()

def plot_catplot(axes, stat, uncertainty_metric, rows, cols, inference_mode, catplot_type):
    import seaborn as sns


    for i, selection_metric in enumerate(stat.keys()):
        for j, test_MS in enumerate(stat[selection_metric].keys()):
            bayesian_stat_df = stat[selection_metric][test_MS]
            prediction_column = "predicted_label_%s" % inference_mode
            bayesian_stat_df["Prediction is correct"] = bayesian_stat_df.apply(lambda row: row["true_label"] == row[prediction_column], axis=1)
            arguments={"data": bayesian_stat_df, "x":"true_label", "y": uncertainty_metric, "hue": "Prediction is correct", "palette":"Set2", "ax":axes[j][i], "hue_order":[True, False]}
            
            if catplot_type=="violinplot":
                arguments["split"]=True
                arguments["scale"]="count"
                
            if catplot_type=="stripplot":
                arguments["dodge"]=True
                arguments["size"]=4
                arguments["linewidth"]=1
            getattr(sns, catplot_type)(**arguments)

    # annotate(axes, cols, rows)

def plot_uncertainty_catplot(model_params, stat,  uncertainty_metric, inference_mode="from_mode", saved_file_path=None, results=None, catplot_type="swarmplot"):
    import matplotlib

    # font = {
    #     # 'family': 'normal',
    #         'weight': 'bold',
    #         'size': 18}
    #
    # matplotlib.rc('font', **font)

    cols = [selection_metric.replace("_", " ") for selection_metric in stat.keys()]
    rows = [test_MS.replace("_", " ") for test_MS in stat[list((stat.keys()))[0]].keys()]

    str_suptitle = "Params: "
    for i, line in enumerate(readable_params):
        str_suptitle += line + ': ' + str(model_params[line]) + "; "

    num_rows=len(rows)+1 if results else len(rows)
    fig, axes = plt.subplots(num_rows, len(cols), figsize=(int(12 * len(cols)), int(9 * num_rows)), sharey="row")
    plot_catplot(axes, stat,uncertainty_metric, rows, cols, inference_mode, catplot_type)
    if results:
        for k, mode in enumerate(results.keys()):
            plot_bar_plots(axes[-1][k], results[mode], mode)

    plt.suptitle(str_suptitle, fontsize="large", fontweight="bold")
    plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05, wspace=0.1, hspace=0.1)
    if saved_file_path is not None:
        plt.savefig(saved_file_path)
    else:
        plt.show()
    plt.close()

