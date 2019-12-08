'''this script for evaluating the model performance:
it will produce confusion matrix, precision, recall, accuracy.
'''

import csv
import multiprocessing
import numpy as np
import os

from itertools import cycle
from sklearn.metrics import (confusion_matrix,
                             precision_recall_curve,
                             average_precision_score)


from plots import plot_confusion_matrix

RESULTS_PATH = 'results/'
PRECISION_RECALL_PLOTS = 'precision_recall_plots/'
MODELS = 'models/'


def _initialize_dir(name):
    try:
        os.mkdir(name)
    except FileExistsError:
        print("Directory already exists...")


def initialize_results_dir(model_name, accuracy, mean_average_precision):
    model_dir = "{}_acc_{}_map_{}".format(model_name, accuracy, mean_average_precision)
    base_path = os.path.join(RESULTS_PATH, model_dir)
    _initialize_dir(base_path)
    _initialize_dir(os.path.join(base_path,
                                 PRECISION_RECALL_PLOTS))
    _initialize_dir(os.path.join(base_path, MODELS))
    return base_path



def _get_average_precisions(transfer_model, x_test, y_test):
    average_precisions = np.zeros(x_test.shape[0])
    for i in range(x_test.shape[0]):
        if i % 100 == 0:
            print('precisions_done_calculating{}'.format(i))
        num = i
        num_retrievable = (np.argmax(y_test[num]) == \
                               np.argmax(y_test, axis=1)).sum()
        # latent_object = latent_model.predict(x_test[num:num+1])
        #####latent_object = latent_space[num: num+1]
        sims, latent_indices = query_latent_space(latent_object,
                                                  latent_space,
                                                  x_test.shape[0])
        ranked_relevant = np.argmax(y_test[num]) ==\
                            np.argmax(y_test[latent_indices], axis=1)

        average_precisions[i] = average_precision(ranked_relevant, num_retrievable)
    return average_precisions


def _save_model_summary(transmodel, path):
    # def myprint(s):
    #     with open(os.path.join(path, 'modelsummary.txt'), 'w') as f:
    #         print(s, file=f)
    with open(os.path.join(path, 'modelsummary.txt'), 'w') as f:
        model.summary(print_fn=lambda x: f.write(x + '\n'))


def _accuracy(eval_model, x_test, y_test):
    y_pred, x_recon = eval_model.predict(x_test)
    test_accuracy = np.sum(np.argmax(y_pred, 1) == np.argmax(y_test, 1))/y_test.shape[0]
    return test_accuracy


def _save_details(path, **kwargs):
    file_path = os.path.join(path, 'details.csv')
    with open(file_path, 'w') as csv_file:
        writer = csv.writer(csv_file)
        for key, value in kwargs.items():
            writer.writerow([key, value])


def save_map_plot(average_precisions, path, suffix=''):
    import matplotlib.pyplot as plt
    mean_average_precision = np.mean(average_precisions)
    plt.hist(average_precisions, bins=10)
    plt.text(.1, 500, 'Mean Average Precision: {:.2%}'.format(mean_average_precision))
    plt.vlines(mean_average_precision, 0, 800)
    plt.title('Mean Average Precision {}'.format(suffix))
    plt.savefig(os.path.join(path, 'mean_average_precision{}.png'.format(suffix)), bbox_inches='tight')


def save_confusion_matrix(y_test, y_pred, target_names, path, figsize=(15, 15), suffix=''):
    import matplotlib.pyplot as plt
    cm = confusion_matrix(np.argmax(y_test, axis=1), np.argmax(y_pred, axis=1))
    plt.figure(figsize=figsize)
    plot_confusion_matrix(cm, target_names, normalize=True, suffix=suffix)
    plt.savefig(os.path.join(path, 'Confusion_matrix{}.png'.format(suffix)), bbox_inches='tight')
    plt.close()


def plot_precision_recall(y_test, y_pred, target_names,
                          path, save=False, show_figs=False,
                          figsize=(7, 8), suffix=''):
    import matplotlib.pyplot as plt
    colors = cycle(['navy', 'turquoise', 'darkorange', 'cornflowerblue', 'teal'])
    precision = dict()
    recall = dict()
    average_precision = dict()

    for i in range(y_pred.shape[1]):
        precision[i], recall[i], _ = precision_recall_curve(y_test[:, i],
                                                            y_pred[:, i])
        average_precision[i] = average_precision_score(y_test[:, i],
                                                       y_pred[:, i])

    # order = sorted(average_precision, key=lambda x: x[1], reverse=True)
    order = list(zip(*sorted(average_precision.items(),
                             key=lambda x: x[1],
                             reverse=True)))[0]

    fig = plt.figure(figsize=figsize)
    lines = []
    labels = []
    fig_count = 0
    count = 0
    # for i, color in zip(range(y_test.shape[1]), colors):
    for idx, i, color in zip(range(1, y_test.shape[1]+1), order, colors):
        l, =plt.plot(recall[i], precision[i], color=color, lw=2)
        lines.append(l)
        labels.append('Precision-recall for class {} (area= {})'\
                          .format(target_names[i], round(average_precision[i], 2)))
    #     print(round(average_precision[i], 2))
        if idx % 5 == 0 and idx != 0:
    #         fig = plt.gcf()
            fig.subplots_adjust(bottom=0.25)
            plt.legend(lines, labels, loc=(0, .18))
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('Recall')
            plt.ylabel('Precision')
            plt.title('Precision-Recall {}'.format(suffix))
            lines = []
            labels = []
            if save:
                plt.savefig(os.path.join(path, PRECISION_RECALL_PLOTS, 'precision_recall {}{}'.format(fig_count, suffix)))
                fig_count += 1
            if show_figs:
                plt.show()
            plt.close()
            plt.figure(figsize=figsize)
        count += 1


def process_results(name: str, eval_model,
                    manipulate_model, x_test, y_test, target_names,
                    **details):
    "Takes all outputs you care about and logs them to results folder"
    latent_model = _make_latent_model(eval_model)
    latent_space = _make_latent_space(latent_model, x_test)
    rotated_about_z = np.rot90(x_test, axes=(1, 2))
    latent_space_rotated = _make_latent_space(latent_model, rotated_about_z)
    def acc_map_metrics(x_test, latent_space):
        accuracy = str(round(_accuracy(eval_model,
                                       x_test,
                                       y_test), 5)).replace('.', '')
        average_precisions = _get_average_precisions(latent_model,
                                                     latent_space,
                                                     x_test, y_test)
        mean_avg_prec = str(round(np.mean(average_precisions),5)).replace('.', '')
        return accuracy, mean_avg_prec, average_precisions
    accuracy, mean_avg_prec, average_precisions = acc_map_metrics(x_test,
                                                                  latent_space)
    rot_accuracy, rot_mean_avg_prec, rot_average_precisions = acc_map_metrics(rotated_about_z,
                                                                              latent_space_rotated)

    dir_path = initialize_results_dir(name, accuracy, mean_avg_prec)
    _save_details(dir_path, accuracy=accuracy,
                  mean_avg_prec=mean_avg_prec,
                  rot_accuracy=rot_accuracy,
                  rot_mean_avg_prec=rot_mean_avg_prec,
                  **details)
