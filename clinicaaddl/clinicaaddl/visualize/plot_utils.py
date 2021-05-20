import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import json
import numpy as np
from matplotlib import cm


class Plots():
    def __init__(self):
        self.description = "plots functionality"

    @staticmethod
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

    @staticmethod
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

    @staticmethod
    def loss_with_results(params, results, history, saved_file_path=None):
        readable_params = ['model', 'data_augmentation', 'batch_size', 'learning_rate', "loss", 'training MS']
        num_figures = len(results.keys())
        fig, axes = plt.subplots(2, num_figures, figsize=(int(12 * num_figures), 18))
        str_suptitle = "Params: "

        for i, line in enumerate(readable_params):
            str_suptitle += line + ': ' + str(params[line]) + "; "

        for k, mode in enumerate(results.keys()):
            Plots.plot_bar_plots(axes[0][k], results[mode], mode)

        Plots.plot_history(axes[1][0], history, mode='loss')
        Plots.plot_history(axes[1][1], history, mode='balanced_accuracy')

        axes[1][2].axis('off')

        plt.suptitle(str_suptitle)
        if saved_file_path is not None:
            plt.savefig(saved_file_path)
        else:
            plt.show()

        plt.close()
