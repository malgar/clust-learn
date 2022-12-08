"""Visualization utils for the classifier module"""
# Author: Miguel Alvarez-Garcia

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import shap

from matplotlib.collections import QuadMesh
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, auc
from .utils import get_shap_importances
from ..utils import savefig

sns.set_style('whitegrid')


def plot_shap_importances(model, X, n_top=7, output_path=None, savefig_kws=None):
    """
    Plots shap importance values, calculated as the combined average of the absolute values of the shap values
    for all classes.

    Parameters
    ----------
    model : `scikit-learn.Estimator`
        Classification model (already trained).
    X : `pandas.DataFrame` or `numpy.ndarray`
        Observations (predictors).
    n_top : int, default=7
        Top n features to be displayed. The importances of the rest are aggregated and displayed under the tag "Rest".
    output_path : str, default=None
        Path to save figure as image.
    savefig_kws : dict, default=None
        Save figure options.
    """
    si = get_shap_importances(model, X)
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


def plot_shap_importances_beeswarm(model, X, class_id, n_top=10, output_path=None, savefig_kws=None):
    """
    Plots a summary of shap values for a specific class of the target variable. This uses shap beeswarm plot
    (https://shap.readthedocs.io/en/latest/example_notebooks/api_examples/plots/beeswarm.html).

    Parameters
    ----------
    model : `scikit-learn.Estimator`
        Classification model (already trained).
    X : `pandas.DataFrame` or `numpy.ndarray`
        Observations (predictors).
    class_id : int
        The class for which to show the SHAP values.
    n_top : int, default=7
        Top n features to be displayed. The importances of the rest are aggregated and displayed under the tag "Rest".
    output_path : str, default=None
        Path to save figure as image.
    savefig_kws : dict, default=None
        Save figure options.
    """
    explainer = shap.Explainer(model)
    shap_values = explainer(X)
    shap.plots.beeswarm(shap_values[:, :, class_id], show=False, max_display=n_top+1)
    plt.title(f'SHAP values summary for class {class_id}', fontsize=13)
    savefig(output_path, savefig_kws)


def plot_confusion_matrix(cf, group_names=None, count=True, percent=True, sum_stats=True, xyticks=True,
                          xyplotlabels=True, figsize=None, cmap='Blues', title=None, output_path=None,
                          savefig_kws=None):
    """
    This function makes a pretty plot of an sklearn Confusion Matrix cf using a Seaborn heatmap visualization.

    Inspired by : https://github.com/DTrimarchi10/confusion_matrix/blob/master/cf_matrix.py

    Parameters
    ---------
    cf : `pandas.DataFrame`
        Confusion matrix to be passed in.
    group_names : list, default=None
        List of strings that represent the labels row by row to be shown in each square.
    count : boolean, default=True
        If True, show the raw number in the confusion matrix.
    percent : boolean, default=True
        If True, show the percentages in the confusion matrix.
    sum_stats : boolean, default=True
        If True, show precision and recall per class, and global accuracy, appended to the matrix.
    xyticks : boolean, default=True
        If True, show x and y ticks.
    xyplotlabels : boolean, default=True
        f True, show 'Observed values' and 'Predicted values' on the figure.
    figsize : tuple, default=None
        Tuple representing the figure size. Default will be the matplotlib default value.
    cmap : `matplotlib.pyplot.cm`, default='Blues'
        Colormap of the values displayed from matplotlib.pyplot.cm. Default is 'Blues'
        See http://matplotlib.org/examples/color/colormaps_reference.html
    title : str, feault=None
        Title for the heatmap.
    output_path : str, default=None
        Path to save figure as image.
    savefig_kws : dict, default=None
        Save figure options.
    """

    # CODE TO GENERATE TEXT INSIDE EACH SQUARE
    blanks = ['' for i in range(cf.size)]

    if group_names and len(group_names) == cf.size:
        group_labels = ["{}\n".format(value) for value in group_names]
    else:
        group_labels = blanks

    flatten = cf.values.flatten()
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
            group_percentages = ["{0:.2%}".format(flatten[i] / np.sum(cf.values)) if (
                        i % cf.shape[0] < cf.shape[0] - 1 and i // cf.shape[0] < cf.shape[0] - 1)
                                 else "" for i in range(len(flatten))]
        else:
            group_percentages = ["{0:.2%}".format(value) for value in cf.values.flatten() / np.sum(cf.values)]
    else:
        group_percentages = blanks

    box_labels = [f"{v1}{v2}{v3}".strip() for v1, v2, v3 in zip(group_labels, group_counts, group_percentages)]
    box_labels = np.asarray(box_labels).reshape(cf.shape[0], cf.shape[1])

    # Cluster names as tick labels. If precision, recall, and global accuracy are included, no tick label is displayed
    # for them
    xyticklabels = False
    if xyticks:
        if sum_stats:
            xyticklabels = list(map(lambda x: x[1], cf.columns[:-1])) + ['']
        else:
            xyticklabels = list(map(lambda x: x[1], cf.columns))

    # MAKE THE HEATMAP VISUALIZATION
    ax, fig = plt.subplots(figsize=figsize)
    sns.heatmap(cf.values, annot=box_labels, fmt="", cmap=cmap, cbar=False, xticklabels=xyticklabels,
                yticklabels=xyticklabels)

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
    """
    Plots ROC curve for every class.

    Parameters
    ---------
    X : `pandas.DataFrame` or `numpy.ndarray`
        Predictor values.
    y : `pandas.Series` or `numpy.array`
        Target values.
    model : `scikit-learn.Estimator`
        Classification model (already trained).
    output_path : str, default=None
        Path to save figure as image.
    savefig_kws : dict, default=None
        Save figure options.
    """
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
