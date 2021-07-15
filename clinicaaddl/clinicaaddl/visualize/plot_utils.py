import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import json
import numpy as np
from matplotlib import cm


def plot_history_ax(ax, history, mode, aggregation_type):
    import seaborn as sns
    def find_best(arr, ismin=True):
        arr = np.array(arr)
        if ismin:
            best_loss_idx_train = np.where(arr == np.amin(arr))[0][0]
        else:
            best_loss_idx_train = np.where(arr == np.amax(arr))[0][0]
        return best_loss_idx_train, arr[best_loss_idx_train]

    sns.lineplot(data=history, ax=ax, x="epoch", y=mode + "_train", label='train', legend="brief")
    sns.lineplot(data=history, ax=ax, x="epoch", y=mode + "_valid", label='validation', legend="brief")
    if aggregation_type is not "all":
        idx, val = find_best(history[mode + "_valid"], mode == 'loss')
        ax.plot(idx, val, 'o', color='black')

    if mode == 'loss':
        ax.set_ylim(bottom=-0.001, top=0.5)
    if mode == 'balanced_accuracy':
        ax.set_ylim(bottom=-0.001, top=1.1)
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels,
              bbox_to_anchor=(0.5, -0.14),
              loc='upper center',
              ncol=2, fontsize=18
              )
    title=mode
    if "_" in title:
        title=title.replace("_", " ")
    # ax.set_title(title)
    ax.tick_params(axis="x", labelsize=18)
    ax.tick_params(axis="y", labelsize=18)
    plt.xlabel("epoch", fontsize=18)
    plt.ylabel(title, fontsize=18)


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
                # _y = p.get_y() + p.get_height() +ax.get_ylim()[-1]*factor
                _y = p.get_y() + ax.get_ylim()[-1] * factor

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
    sns.barplot(data=results,hue="metric", x="mode", y="value", ax=ax, palette="Paired", edgecolor="gray",  alpha=.95, linewidth=1.5)

    ax.set_ylim(bottom=-0.001, top=1.1)
    show_values_on_bars(ax)
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles[len(columns):], labels[len(columns):],
              bbox_to_anchor=(0.5, -0.1),
              loc='upper center',
              ncol=3
              )



def plot_results_agg_ax(ax, results, columns):

    def reshape_results(results, columns):
        reshaped_columns = ["mode", "metric", "value", "fold"]
        reshaped_df = pd.DataFrame(columns=reshaped_columns)
        for col in list(results[columns].columns):
            for idx, row in results.iterrows():
                new_row = [[row["mode"], col, row[col], row["fold"]]]
                row_df = pd.DataFrame(new_row, columns=reshaped_columns)
                reshaped_df = pd.concat([reshaped_df, row_df], axis=0)

        return reshaped_df

    def show_values_on_bars(axs):
        def _show_on_single_plot(ax, factor=0.025):
            for p in ax.patches:
                _x = p.get_x() + p.get_width() / 2
                # _y = p.get_y() + p.get_height() +ax.get_ylim()[-1]*factor
                _y = p.get_y() +ax.get_ylim()[-1]*factor

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
    results['mode'].replace({'test_1.5T': 'test 1.5T','test_3T':'test 3T'}, inplace=True)

    sns.swarmplot(data=results,hue="metric", x="mode", y="value", ax=ax, palette=sns.color_palette("Paired"),edgecolor="black", alpha=1., linewidth=1.0, dodge=True)
    results_avg=results.groupby(["mode", "metric"], as_index=False, sort=False).agg(np.mean)
    sns.barplot(data=results_avg,hue="metric", x="mode", y="value", ax=ax, palette=sns.color_palette("Paired"),edgecolor="gray",  alpha=.95, linewidth=1.5)
    show_values_on_bars(ax)
    ax.set_ylim(bottom=-0.001, top=1.05)
    handles, labels = ax.get_legend_handles_labels()
    legend=ax.legend(handles[len(columns):], labels[len(columns):],
                  bbox_to_anchor=(0.5, -0.06),
                  loc='upper center',
                  ncol=2, fontsize=18
                  )
    legend.get_frame().set_alpha(None)
    ax.tick_params(axis="x", labelsize=18)
    ax.tick_params(axis="y", labelsize=18)
    plt.ylabel("")


def plot_catplot_ax(ax, data, uncertainty_metric, inference_mode, catplot_type):
    import seaborn as sns
    


    prediction_column = "predicted_label_from_%s" %inference_mode
    data["Prediction is correct"] = data.apply(
        lambda row: row["true_label"] == row[prediction_column], axis=1)
    
    data['true_label'].replace({0: 'CN',1:'AD'}, inplace=True)
    data['Prediction is correct'].replace({True: 'Correct',False:'Incorrect'}, inplace=True)

    arguments = {"data": data, "x": "true_label", "y": uncertainty_metric,
                 "hue": "Prediction is correct", "palette": "Set2", "ax": ax,
                 # "hue_order": [True, False],
                 "order":["CN", "AD"]}

    if catplot_type == "violinplot":
        arguments["split"] = True
        arguments["scale"] = "count"
        arguments["cut"] = 0

    if catplot_type == "stripplot":
        arguments["dodge"] = True
        arguments["size"] = 4
        arguments["linewidth"] = 1
    getattr(sns, catplot_type)(**arguments)
    
    handles, labels = ax.get_legend_handles_labels()
    legend=ax.legend(handles, labels,
              bbox_to_anchor=(0.5, -0.06),
              loc='upper center',
              ncol=2, fontsize=18
              )

    legend.get_frame().set_alpha(None)
    ax.tick_params(axis="x", labelsize=18)
    ax.tick_params(axis="y", labelsize=18)
    uncertainty_metric_label=uncertainty_metric.replace("_", " ")
    plt.ylabel(uncertainty_metric_label, fontsize=18)


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
#     min_ylim=min(ax.get_ylim()[0] for ax in axes)
#     max_ylim=max(ax.get_ylim()[1] for ax in axes)
    min_ylim=-0.01
    max_ylim=0.3
    for ax in axes:
        ax.set_ylim(bottom=min_ylim, top=max_ylim)

#
# def plot_hist(axes, stat, uncertainty_metric, rows, cols, separate_by_labels):
#     import seaborn as sns
#     import numpy as np
#
#     xlim_list={"left_limit":[], "right_limit":[]}
#
#     for i, selection_metric in enumerate(stat.keys()):
#         for j, test_MS in enumerate(stat[selection_metric].keys()):
#             st = stat[selection_metric][test_MS]
#             sns.histplot(data=st, x=st[uncertainty_metric], hue=st.true_label.values if separate_by_labels else None,
#                          ax=axes[j][i], stat="probability", bins=10)
#             xlim_list["left_limit"].append(min(axes[j][i].get_xlim()))
#             xlim_list["right_limit"].append(max(axes[j][i].get_xlim()))
#
#     #set_xlim for all histogram plots
#     xlim_left,xlim_right =np.min(xlim_list["left_limit"]), np.max(xlim_list["right_limit"])
#     for i, row in enumerate(rows):
#         for j, col in enumerate(cols):
#             axes[i][j].set_xlim(left=xlim_left, right=xlim_right)
#
#     annotate(axes, cols, rows)






