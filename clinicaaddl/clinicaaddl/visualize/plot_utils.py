import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import json
import numpy as np
from matplotlib import cm

readable_params = ['model', 'data_augmentation', 'batch_size', 'learning_rate', "loss", 'training MS']


def plot_history(ax, history, mode):
    def find_best(arr, ismin=True):
        arr = np.array(arr)
        if ismin:
            best_loss_idx_train = np.where(arr == np.amin(arr))[0][0]
        else:
            best_loss_idx_train = np.where(arr == np.amax(arr))[0][0]
        return best_loss_idx_train, arr[best_loss_idx_train]

    ax.plot(history["epoch"], history[mode + "_train"], 'black', lw=1, label='train ' + mode)
    ax.plot(history["epoch"], history[mode + "_valid"], 'red', lw=1, label='validation ' + mode)

    idx, val = find_best(history[mode + "_valid"], mode == 'loss')
    ax.plot(idx, val, 'o', color='black')

    #     ax.legend()
    ax.legend(bbox_to_anchor=(0.5, -0.15), loc='lower center',
              ncol=2, fontsize='large')

    if mode == 'loss':
        ax.set_ylim(bottom=-0.001, top=0.5)
    if mode == 'balanced_accuracy':
        ax.set_ylim(bottom=-0.001, top=1.1)
    ax.set_title(mode)

def plot_bar_plots(ax, results, mode):
    def autolabel(ax, rects):
        for rect in rects:
            height = rect.get_height()
            ax.text(rect.get_x() + rect.get_width() / 2., 1.02 * height,
                    '%.3f' % (height),
                    ha='center', va='bottom')

    def reshape_results(results):
        metrics = {}
        for key in results[list(results.keys())[0]].keys():
            metrics[key] = []
        for mode in results.keys():
            for metric in results[mode].keys():
                metrics[metric].append(results[mode][metric][0])
        return metrics

    ax.set_ylim(bottom=-0.001, top=1.1)
    results_transposed = reshape_results(results)
    N = len(results.keys())

    ind = np.arange(N)
    width = 0.2
    width_ratio = 0.9
    #     ax.set_prop_cycle('color', Pastel1_4.mpl_colors)
    ax.set_prop_cycle('color', cm.get_cmap('Paired').colors)

    for i, key in enumerate(results_transposed.keys()):
        autolabel(ax, ax.bar(ind + width * i, results_transposed[key], width * width_ratio, label=key.capitalize(),
                             edgecolor='dimgrey'))

    xstips_position = ind + width

    xsticklabels = [disease_type for disease_type in results.keys()]
    ax.set_xticklabels(xsticklabels)
    ax.set_xticks(xstips_position)
    ax.legend(bbox_to_anchor=(0.5, -0.15), loc='lower center',
              ncol=3, fontsize='large')
    ax.set_title("Model from: " + mode)

def barplots_with_loss(params, results, history, saved_file_path=None):
    num_figures = len(results.keys())
    fig, axes = plt.subplots(2, num_figures, figsize=(int(12 * num_figures), 18))
    str_suptitle = "Params: "

    for i, line in enumerate(readable_params):
        str_suptitle += line + ': ' + str(params[line]) + "; "

    for k, mode in enumerate(results.keys()):
        plot_bar_plots(axes[0][k], results[mode], mode)

    plot_history(axes[1][0], history, mode='loss')
    plot_history(axes[1][1], history, mode='balanced_accuracy')

    axes[1][2].axis('off')

    plt.suptitle(str_suptitle)
    plt.subplots_adjust(left=None, right=None, top=None, bottom=None, wspace=None, hspace=None)

    if saved_file_path is not None:
        plt.savefig(saved_file_path)
    else:
        plt.show()
    plt.close()




def plot_hist(axes, stat, uncertainty_metric, rows, cols, separate_by_labels):
    import seaborn as sns
    import numpy as np

    def annotate(axes, cols, rows):
        for ax, col in zip(axes[0], cols):
            ax.annotate(col, xy=(0.5, 1), xytext=(0, 5),
                        xycoords='axes fraction', textcoords='offset points',
                        fontsize=20, ha='center', va='baseline')

        for ax, row in zip(axes[:, 0], rows):
            ax.annotate(row, xy=(0, 0.5), xytext=(-ax.yaxis.labelpad - 5, 0),
                        xycoords=ax.yaxis.label, textcoords='offset points',
                        fontsize=15, ha='right', va='center')
    xlim_list={"left_limit":[], "right_limit":[]}

    for i, selection_metric in enumerate(stat.keys()):
        for j, test_MS in enumerate(stat[selection_metric].keys()):
            st = stat[selection_metric][test_MS]
            sns.histplot(data=st, x=st[uncertainty_metric], hue=st.true_label.values if separate_by_labels else None,
                         ax=axes[j][i], stat="probability", bins=100)
            xlim_list["left_limit"].append(min(axes[j][i].get_xlim()))
            xlim_list["right_limit"].append(max(axes[j][i].get_xlim()))


    #set_xlim for all histogram plots
    xlim_left,xlim_right =np.min(xlim_list["left_limit"]), np.max(xlim_list["right_limit"])
    for i, row in enumerate(rows):
        for j, col in enumerate(cols):
            axes[i][j].set_xlim(left=xlim_left, right=xlim_right)

    annotate(axes, cols, rows)


def plot_uncertainty_dist(model_params, stat,  uncertainty_metric, separate_by_labels=False, saved_file_path=None, results=None):
    import seaborn as sns

    cols = [selection_metric.replace("_", " ") for selection_metric in stat.keys()]
    rows = [test_MS.replace("_", " ") for test_MS in stat[list((stat.keys()))[0]].keys()]

    str_suptitle = "Params: "

    for i, line in enumerate(readable_params):
        str_suptitle += line + ': ' + str(model_params[line]) + "; "

    num_rows=len(rows)+1 if results else len(rows)
    fig, axes = plt.subplots(num_rows, len(cols), figsize=(int(18 * len(cols)), int(12 * num_rows)), sharey="row")

    plot_hist(axes, stat,uncertainty_metric, rows, cols, separate_by_labels)

    if results:
        for k, mode in enumerate(results.keys()):
            plot_bar_plots(axes[-1][k], results[mode], mode)

    plt.suptitle(str_suptitle)
    plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05, wspace=0.05, hspace=0.05)
    if saved_file_path is not None:
        plt.savefig(saved_file_path)
    else:
        plt.show()
    plt.close()



# def scatter_variance_per_class(model_params, stat, plot_filename):
#     import seaborn as sns
#     st = stat["best_loss"]["test_1.5T"]
#     x = np.array(st.class_variance.values.tolist())[:, 0]
#     y = np.array(st.class_variance.values.tolist())[:, 1]
#     sns.scatterplot(data=st, x=x, y=y, hue=st.true_label.values)
#     plt.show()
#     print("smth")