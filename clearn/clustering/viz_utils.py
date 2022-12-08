"""Utils for visualization"""
# Author: Miguel Alvarez-Garcia

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from kneed import KneeLocator
from .table_utils import compare_cluster_means_to_global_means
from ..utils import get_axis, plot_optimal_normalized_elbow, savefig


sns.set_style('whitegrid')

CARTO_COLORS = ['#7F3C8D', '#11A579', '#3969AC', '#F2B701', '#E73F74', '#80BA5A', '#E68310',  '#008695', '#CF1C90',
                '#f97b72', '#4b4b8f', '#A5AA99']


def plot_score_comparison(scores, cluster_range, metric_name='Weighted sum of squared distances', output_path=None,
                          savefig_kws=None):
    """
    Plots the comparison in performance between the different clustering algorithms.

    Parameters
    ----------
    scores : dict
        Dictionary <algorithm, list of scores>
    cluster_range : [min (int), max (int))
        Range of number of clusters computed. This will be displayed on the x-axis.
    metric_name : str, default='Weighted sum of squared distances'
        Name of the metric used for comparison. This will be displayed on the y-label.
        Default is 'Weighted sum of squared distances', which corresponds to inertia.
    output_path : str, default=None
        Path to save figure as image.
    savefig_kws : dict, default=None
        Save figure options.
    """
    plt.figure(figsize=(10, 5))
    i = 0
    for algorithm in scores:
        plt.plot(scores[algorithm], label=algorithm, color=CARTO_COLORS[i])
        i += 1

    plt.xlabel('Number of clusters', fontsize=12, labelpad=15)
    plt.ylabel(metric_name, fontsize=12, labelpad=15)
    plt.xticks(ticks=range(len(list(scores.values())[0])), labels=range(*cluster_range))
    plt.tight_layout()
    plt.legend(fontsize=12, title='Algorithm', title_fontsize=13, labelspacing=0.5)

    savefig(output_path=output_path, savefig_kws=savefig_kws)


def plot_optimal_components_normalized(scores, max_clusters, metric_name, output_path=None, savefig_kws=None):
    """
    Plots the normalized curve used for computing the optimal number of clusters.

    Parameters
    ----------
    scores : dict
        Dictionary <algorithm, list of scores>
    max_clusters : int
        Maximum number of clusters allowed.
    metric_name : str, default='Weighted sum of squared distances'
        Name of the metric used for comparison. This will be displayed on the y-label.
        Default is 'Weighted sum of squared distances', which corresponds to inertia.
    output_path : str, default=None
        Path to save figure as image.
    savefig_kws : dict, default=None
        Save figure options.
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    kl = KneeLocator(x=range(1, max_clusters + 1), y=scores, curve='convex', direction='decreasing')
    plot_optimal_normalized_elbow(scores, kl, ax, optimal_label='Optimal number of clusters',
                                  xlabel='Number of clusters', ylabel=f'Normalized {metric_name}')
    savefig(output_path=output_path, savefig_kws=savefig_kws)


def plot_clustercount(df, output_path=None, savefig_kws=None):
    """
    Plots a bar plot with cluster counts.

    Parameters
    ----------
    df : `pandas.DataFrame`
        DataFrame containing at least a column named 'cluster_cat' with the cluster labels.
    output_path : str, default=None
        Path to save figure as image.
    savefig_kws : dict, default=None
        Save figure options.
    """
    plt.figure(figsize=(df['cluster_cat'].nunique(), 5))
    sns.countplot(x='cluster_cat', data=df, color='#332288', alpha=0.9, order=np.sort(df['cluster_cat'].unique()))
    # plt.xticks(rotation=30)
    plt.ylabel('count', fontsize=12, labelpad=10)
    plt.xlabel('clusters', fontsize=12, labelpad=10)
    plt.tight_layout(pad=2)
    savefig(output_path=output_path, savefig_kws=savefig_kws)


def plot_cluster_means_to_global_means_comparison(df, dimensions, xlabel=None, ylabel=None,
                                                  levels=[-0.50, -0.32, -0.17, -0.05, 0.05, 0.17, 0.32, 0.50],
                                                  data_standardized=False, output_path=None, savefig_kws=None):
    """
    Plots the normalized curve used for computing the optimal number of clusters.

    Parameters
    ----------
    df : `pandas.DataFrame`
        DataFrame containing the variables used for clustering.
    dimensions : list
        List of variables of interest.
        *Note these must be internal variables, ie, variables used for clustering*
    xlabel : str, default=None
        x-label name/description.
    ylabel : str, default=None
        y-label name/description.
    levels : list or `numpy.array`
        Values to be used as cuts for color intensity.
        Default values: [-0.50, -0.32, -0.17, -0.05, 0.05, 0.17, 0.32, 0.50]
    data_standardized : bool, default=False
        If data are standardized, comparison to global mean is based solely on the mean per cluster.
    output_path : str, default=None
        Path to save figure as image.
    savefig_kws : dict, default=None
        Save figure options.
    """
    df_diff = compare_cluster_means_to_global_means(df, dimensions, data_standardized=data_standardized)
    colors = sns.color_palette("BrBG", n_colors=9)
    cmap, norm = matplotlib.colors.from_levels_and_colors(levels, colors, extend="both")
    width = min(len(dimensions), 20)
    height = min(df['cluster'].nunique(), 8)
    fig, ax = plt.subplots(figsize=(width, height))
    im = ax.imshow(df_diff[dimensions].values, cmap=cmap, norm=norm)
    ax.set(xticks=range(len(dimensions)), yticks=range(df_diff.shape[0]),
           xticklabels=list(map(str.upper, dimensions)), yticklabels=df_diff['cluster'])
    ax.tick_params(axis='x', rotation=40, labelsize=10)
    ax.tick_params(axis='y', labelsize=10)
    ax.set_xlabel('' if xlabel is None else xlabel, fontsize=12, weight='bold', labelpad=15)
    ax.set_ylabel('' if ylabel is None else ylabel, fontsize=12, weight='bold', labelpad=15)
    for i in range(len(df_diff['cluster'].unique())):
        for j in range(len(dimensions)):
            val = df_diff.loc[i, dimensions[j]]
            val_str = '{:.2f}'.format(val)
            if val < 0:
                val_str = '- ' + '{:.2f}'.format(-val)

            text = ax.text(j, i, val_str,
                           ha="center", va="center", color="black", fontsize=11, fontweight='ultralight',
                           fontstretch='ultra-expanded')

    # Turns off grid on the left axis
    ax.grid(False)
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    savefig(output_path=output_path, savefig_kws=savefig_kws)


def plot_distribution_by_cluster(df, cluster_labels, xlabel=None, ylabel=None, sharex=True, sharey=True,
                                 output_path=None, savefig_kws=None):
    """
    Plots the violin plots per cluster and *continuous* variables of interest to understand differences in their
    distributions by cluster.

    Parameters
    ----------
    df : `pandas.DataFrame`
        DataFrame containing the variables used for clustering.
    cluster_labels : `numpy.array` or list
        Array with cluster labels.
        *Note this array should have the same length as df and observations be in the same order*.
    xlabel : str, default=None
        x-label name/description.
    ylabel : str, default=None
        y-label name/description.
    sharex : bool, default=True
        If True, all subplots share the x-axis.
    sharey : bool, default=True
        If True, all subplots share the y-axis.
    output_path : str, default=None
        Path to save figure as image.
    savefig_kws : dict, default=None
        Save figure options.
    """
    nclusters = len(np.unique(cluster_labels))
    nvars = df.shape[1]
    ncols = max(1, min(nvars, 18//nclusters))
    if ncols > 3 and nvars % ncols > 0:
        if nvars % 3 == 0:
            ncols = 3
        elif nvars % 2 == 0:
            ncols=2

    nrows = nvars // ncols + (nvars % ncols > 0)
    fig, axs = plt.subplots(nrows, ncols, figsize=(max(nclusters * ncols, 9), 5 * nrows), sharex=sharex, sharey=sharey)

    i = 0
    for col in df.columns:
        ax = get_axis(i, axs, ncols, nrows)
        sns.violinplot(y=df[col], x=cluster_labels, linewidth=1, ax=ax)
        plt.setp(ax.collections, alpha=.4)
        sns.boxplot(y=df[col], x=cluster_labels, width=0.2, linewidth=1, color='grey', ax=ax)
        sns.stripplot(y=df[col], x=cluster_labels, alpha=0.7, size=3, ax=ax)
        ax.set_ylabel(col if ylabel is None else ylabel, fontsize=12, labelpad=15)
        if i // ncols == nrows-1:
            ax.set_xlabel('cluster' if xlabel is None else xlabel, fontsize=12, labelpad=15)
        i += 1

    while i < ncols * nrows:
        ax = get_axis(i, axs, ncols, nrows)
        ax.axis('off')
        i += 1

    fig.tight_layout(pad=2)
    savefig(output_path=output_path, savefig_kws=savefig_kws)


def plot_clusters_2D(x, y, hue, df, style_kwargs=dict(), output_path=None, savefig_kws=None):
    """
    Plots two 2D plots:
     - A scatter plot styled by the categorical variable `hue`.
     - A 2D plot comparing cluster centroids and optionally the density area.

    Parameters
    ----------
    x : `numpy.array` or list
        x-coordinate data.
    y : `numpy.array` or list
        y-coordinate data.
    hue : `numpy.array` or list
        Array with categorical values to be used for color styling.
    df : `pandas.DataFrame`
        DataFrame containing the data.
    style_kwargs : dict, default=empty dict
        Dictionary with optional styling parameters.
        List of parameters:
         - palette : matplotlib palette to be used. default='gnuplot'
         - vline_color : color to be used for vertical line (used for plotting x mean value). default='#11A579'
         - hline_color : color to be used for horizontal line (used for plotting y mean value). default='#332288'
         - kdeplot : boolean to display density area of points (using seabonr.kdeplot). default=True
    output_path : str, default=None
        Path to save figure as image.
    savefig_kws : dict, default=None
        Save figure options.
    """
    # Style params
    palette = 'gnuplot'
    if style_kwargs.get('palette'):
        palette = style_kwargs.get('palette')

    vline_color = '#11A579'
    if style_kwargs.get('vline_color'):
        vline_color = style_kwargs.get('vline_color')

    hline_color = '#332288'
    if style_kwargs.get('hline_color'):
        vline_color = style_kwargs.get('hline_color')

    kdeplot = True
    if style_kwargs.get('kdeplot') is not None:
        kdeplot = style_kwargs.get('kdeplot')

    fig, axs = plt.subplots(1, 2, figsize=(14, 5), sharey=True, sharex=True)

    x_range = df[x].max() - df[x].min()
    xmin = df[x].min() - x_range * 0.05
    xmax = df[x].max() + x_range * 0.05
    y_range = df[y].max() - df[y].min()
    ymin = df[y].min() - y_range * 0.05
    ymax = df[y].max() + y_range * 0.05

    # Left-hand side plot: Scatter plot colored by cluster category
    sns.scatterplot(x=x, y=y, hue=hue, data=df.sort_values(hue), alpha=0.3, palette=palette, linewidth=0, ax=axs[0])
    axs[0].vlines(df[x].mean(), ymin=ymin, ymax=ymax, color=vline_color, linewidth=1.15, linestyles='--', label=f'Mean {x}')
    axs[0].hlines(df[y].mean(), xmin=xmin, xmax=xmax, color=hline_color, linewidth=1.15, linestyles='--',
                  label=f'Mean {y}')
    axs[0].set_xlabel(x, fontsize=12)
    axs[0].set_ylabel(y, fontsize=12)
    axs[0].set_title('Scatter plot by cluster', fontsize=13)
    axs[0].set_xlim(xmin, xmax)
    axs[0].set_ylim(ymin, ymax)

    # Right-hand side plot: Cluster centroids with optional kernel density area
    sns.scatterplot(x=x, y=y, hue=hue, data=df.groupby(hue).mean().reset_index(),
                    alpha=1, palette=palette, linewidth=0, marker='X', s=100, ax=axs[1])

    if kdeplot:
        sns.kdeplot(x=x, y=y, hue=hue, data=df.sort_values(hue), levels=1, alpha=0.2, palette=palette,
                    ax=axs[1])

    axs[1].vlines(df[x].mean(), ymin=ymin, ymax=ymax, color=vline_color, linewidth=1, linestyles='--',
                  label=f'Mean {x}')
    axs[1].hlines(df[y].mean(), xmin=xmin, xmax=xmax, color=hline_color, linewidth=1, linestyles='--',
                  label=f'Mean {y}')
    axs[1].set_xlabel(x, fontsize=12)
    axs[1].set_ylabel(y, fontsize=12)
    axs[1].set_title('Cluster centroids', fontsize=13)

    axs[0].legend(fontsize=11, title='', title_fontsize=12, labelspacing=0.5,
                  loc=(0.93, 0.5 - 0.167 * (df[hue].nunique() // 4)))
    axs[1].legend(fontsize=11, title='', title_fontsize=12, labelspacing=0.5,
                  loc=(0.93, 0.5 - 0.167 * (df[hue].nunique() // 4)))

    fig.tight_layout(pad=2)
    savefig(output_path=output_path, savefig_kws=savefig_kws)


def plot_cat_distribution_by_cluster(ct, cat_label=None, cluster_label=None, output_path=None, savefig_kws=None):
    """
    Plots the relative contingency table of the clusters with a categorical variable as a stacked bar plot.

    Parameters
    ----------
    ct : `pandas.DataFrame`
        DataFrame with the relative contingency table.
    cat_label : str, default=None
        Name/Description of the categorical variable to be displayed.
    cluster_label : str, default=None
        Name/Description of the cluster variable to be displayed.
    output_path : str, default=None
        Path to save figure as image.
    savefig_kws : dict, default=None
        Save figure options.
    """
    plt.figure(figsize=(11, 0.625 * len(ct.index)))
    colors = sns.color_palette("YlGnBu", n_colors=len(ct.columns))
    left = np.array([0] * len(ct.index))

    i = 0
    for col in ct.columns:
        widths = ct[col].values
        plt.barh(ct.index, widths, left=left, label=col, color=colors[i], height=0.7)

        xcenters = left + widths / 2
        for y, (x, w) in enumerate(zip(xcenters, widths)):
            if w > 0.05:
                color = '#737373' if i < len(ct.columns)//2 else '#d9d9d9'
                plt.text(x, y, f'{str(np.round(w * 100, 1))}%', ha='center', va='center', color=color, fontsize=12,
                         weight='light')

        left = left + ct[col].values
        i += 1

    ncol = 5
    if len(ct.columns) % ncol > 0:
        if len(ct.columns) % 6 == 0:
            ncol = 6
        elif len(ct.columns) % 4 == 0:
            ncol = 4

    plt.gca().invert_yaxis()
    plt.legend(ncol=ncol, loc='lower center', bbox_to_anchor=(0.5, 1), fontsize=12, title=cat_label,
               title_fontsize=13)
    plt.ylabel(cluster_label, fontsize=12, weight='bold', labelpad=15)
    plt.yticks(ticks=range(len(ct.index)), labels=list(ct.index), fontsize=11)
    plt.xticks([])
    plt.xlim(0, 1)

    plt.tight_layout()
    savefig(output_path=output_path, savefig_kws=savefig_kws)
