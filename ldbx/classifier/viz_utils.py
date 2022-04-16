# Visualization utils for the classifier module

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import utils

from matplotlib.collections import QuadMesh
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, auc
from ..utils import *

sns.set_style('whitegrid')


def plot_shap_importances(model, X, n_top=7, output_path=None, savefig_kws=None):
    si = utils.shap_importances(model, X)
    low_imp = si.loc[n_top:]
    si = si.loc[:n_top - 1]
    si = si.append(pd.DataFrame({'variable_name': ['Rest'], 'shap_importance': [low_imp['shap_importance'].sum()]}),
                   ignore_index=True)
    fig, ax = plt.subplots(figsize=(10, 0.675 * si.shape[0]))
    ax = sns.barplot(x='shap_importance', y='variable_name', data=si, color='#ff0051')
    ax.bar_label(ax.containers[0], padding=5, fmt='%.4f', fontsize=10)
    ax.set_xlabel('mean(|SHAP values|)', fontsize=12)
    ax.set_ylabel('')
    fig.tight_layout()
    savefig(output_path, savefig_kws)


# TODO: Document
def plot_confusion_matrix(cf, group_names=None, categories='auto', count=True, percent=True, sum_stats=True,
                          xyticks=True, xyplotlabels=True, figsize=None, cmap='Blues', title=None,
                          output_path=None, savefig_kws=None):
    """
    This function will make a pretty plot of an sklearn Confusion Matrix cm using a Seaborn heatmap visualization.
    Arguments
    ---------
    cf:            confusion matrix to be passed in
    group_names:   List of strings that represent the labels row by row to be shown in each square.
    categories:    List of strings containing the categories to be displayed on the x,y axis. Default is 'auto'
    count:         If True, show the raw number in the confusion matrix. Default is True.
    cbar:          If True, show the color bar. The cbar values are based off the values in the confusion matrix.
                   Default is True.
    xyticks:       If True, show x and y ticks. Default is True.
    xyplotlabels:  If True, show 'True Label' and 'Predicted Label' on the figure. Default is True.
    figsize:       Tuple representing the figure size. Default will be the matplotlib rcParams value.
    cmap:          Colormap of the values displayed from matplotlib.pyplot.cm. Default is 'Blues'
                   See http://matplotlib.org/examples/color/colormaps_reference.html

    title:         Title for the heatmap. Default is None.
    """

    # CODE TO GENERATE TEXT INSIDE EACH SQUARE
    blanks = ['' for i in range(cf.size)]

    if group_names and len(group_names) == cf.size:
        group_labels = ["{}\n".format(value) for value in group_names]
    else:
        group_labels = blanks

    flatten = cf.flatten()
    if count:
        # Format of totals is different
        if sum_stats:
            group_counts = ["{0:0.0f}\n".format(flatten[i]) if (
                        i % cf.shape[0] < cf.shape[0] - 1 and i // cf.shape[0] < cf.shape[0] - 1)
                            else "{0:.2%}".format(flatten[i]) for i in range(len(flatten))]
        else:
            group_counts = ["{0:0.0f}\n".format(value) for value in flatten]
    else:
        group_counts = blanks

    if percent:
        # Format of totals is different
        if sum_stats:
            group_percentages = ["{0:.2%}".format(flatten[i] / np.sum(cf)) if (
                        i % cf.shape[0] < cf.shape[0] - 1 and i // cf.shape[0] < cf.shape[0] - 1)
                                 else "" for i in range(len(flatten))]
        else:
            group_percentages = ["{0:.2%}".format(value) for value in cf.flatten() / np.sum(cf)]
    else:
        group_percentages = blanks

    box_labels = [f"{v1}{v2}{v3}".strip() for v1, v2, v3 in zip(group_labels, group_counts, group_percentages)]
    box_labels = np.asarray(box_labels).reshape(cf.shape[0], cf.shape[1])

    if not xyticks:
        # Do not show categories if xyticks is False
        categories = False

    # MAKE THE HEATMAP VISUALIZATION
    ax, fig = plt.subplots(figsize=figsize)
    sns.heatmap(cf, annot=box_labels, fmt="", cmap=cmap, cbar=False, xticklabels=categories, yticklabels=categories)

    # Remove colors from totals
    quadmesh = ax.findobj(QuadMesh)[0]
    facecolors = quadmesh.get_facecolors()
    # make colors of the last column white
    facecolors[np.arange(cf.shape[0] - 1, cf.size, cf.shape[0])] = np.array([0, 0, 0, 0.05])
    facecolors[np.arange(cf.size - cf.shape[0], cf.size)] = np.array([0, 0, 0, 0.05])
    quadmesh.set_facecolors = facecolors

    if xyplotlabels:
        plt.ylabel('Observed values', fontsize=11)
        plt.xlabel('Predicted values', fontsize=11)

    if title:
        plt.title(title)

    plt.yticks(rotation=0)
    plt.tight_layout(pad=2)
    savefig(output_path, savefig_kws)


def plot_roc_curves(X, y, model, output_path=None, savefig_kws=None):
    classes = np.sort(np.unique(y))
    y_score = model.predict_proba(X)
    y_test_b = label_binarize(y, classes=classes)

    # Compute ROC curve and AUC for every class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(len(classes)):
        fpr[i], tpr[i], _ = roc_curve(y_test_b[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(y_test_b.ravel(), y_score.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    # Plot curves
    ncols = 3
    if len(classes) % 3 > 0 and len(classes) % 2 == 0:
        ncols = 2
    nrows = int(np.ceil(len(classes) / ncols))

    fig, axs = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3 * nrows))
    for i in range(len(classes)):
        axs[i // ncols, i % ncols].plot(fpr[i], tpr[i], color='#11A579',
                                        label='ROC curve (area = %0.4f)' % roc_auc[i])
        axs[i // ncols, i % ncols].plot([0, 1], [0, 1], color='#7F3C8D', linestyle='--')
        axs[i // ncols, i % ncols].set_xlim([-0.025, 1.025])
        axs[i // ncols, i % ncols].set_ylim([0.0, 1.05])
        axs[i // ncols, i % ncols].set_title(f'Cluster {classes[i]}', fontsize=12, pad=10)
        axs[i // ncols, i % ncols].legend(loc="lower right")

        if i // ncols == nrows - 1:
            axs[i // ncols, i % ncols].set_xlabel('False Positive Rate', fontsize=11, labelpad=10)
        if i % ncols == 0:
            axs[i // ncols, i % ncols].set_ylabel('True Positive Rate', fontsize=11, labelpad=10)

    fig.tight_layout()
    savefig(output_path, savefig_kws)
