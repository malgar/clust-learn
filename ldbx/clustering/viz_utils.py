# Utils for visualization

import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

from kneed import KneeLocator
from table_utils import *
from ..utils import *


sns.set_style('whitegrid')

CARTO_COLORS = ['#7F3C8D', '#11A579', '#3969AC', '#F2B701', '#E73F74', '#80BA5A', '#E68310',  '#008695', '#CF1C90',
                '#f97b72', '#4b4b8f', '#A5AA99']


def plot_score_comparison(scores, cluster_range, metric_name='Weighted sum of squared distances', output_path=None,
                          savefig_kws=None):
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


def plot_optimal_components_normalized(scores, max_clusters, metric_name):
    fig, ax = plt.subplots(figsize=(8, 5))
    kl = KneeLocator(x=range(1, max_clusters + 1), y=scores, curve='convex', direction='decreasing')
    plot_optimal_normalized_elbow(scores, kl, ax, optimal_label='Optimal number of clusters',
                                  xlabel='Number of clusters', ylabel=f'Normalized {metric_name}')


def plot_cluster_means_to_global_means_comparison(df, dimensions, xlabel=None, ylabel=None, output_path=None,
                                                  savefig_kws=None):
    df_diff = compare_cluster_means_to_global_means(df, dimensions)
    colors = sns.color_palette("BrBG", n_colors=9)
    # TODO: These cuts should be passed with some default values
    levels = [-0.50, -0.32, -0.17, -0.05, 0.05, 0.17, 0.32, 0.50]
    cmap, norm = matplotlib.colors.from_levels_and_colors(levels, colors, extend="both")
    fig, ax = plt.subplots(figsize=(20, 8))
    im = ax.imshow(df_diff[dimensions].values, cmap=cmap, norm=norm)
    ax.set(xticks=range(len(dimensions)), yticks=range(df_diff.shape[0]),
           xticklabels=list(map(str.upper, dimensions)), yticklabels=df_diff['cluster_cat'])
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


# TODO: Make corresponding calls from main clustering class
def plot_distribution_comparison_by_cluster(df, cluster_labels, xlabel=None, ylabel=None, output_path=None,
                                            savefig_kws=None):

    nclusters = len(np.unique(cluster_labels))
    nvars = df.shape[0]
    ncols = max(1, min(nvars, 18//nclusters))
    if ncols > 3 and nvars % ncols > 0:
        if nvars % 3 == 0:
            ncols = 3
        elif nvars % 2 == 0:
            ncols=2

    nrows = nvars // ncols + (nvars % ncols > 0)
    fig, axs = plt.subplots(nrows, ncols, figsize=(max(nclusters * ncols, 9), 6 * nrows))

    i = 0
    for col in df.columns:
        ax = get_axis(i, axs, ncols, nrows)
        sns.violinplot(y=df[col], x=cluster_labels, linewidth=1, ax=ax)
        plt.setp(ax.collections, alpha=.4)
        sns.boxplot(y=df[col], x=cluster_labels, width=0.2, linewidth=1, color='grey', ax=ax)
        sns.stripplot(y=df[col], x=cluster_labels, alpha=0.9, size=3)
        ax.set_xlabel('custer' if xlabel is None else xlabel, fontsize=12, weight='bold', labelpad=15)
        ax.set_ylabel(col if ylabel is None else ylabel, fontsize=12, weight='bold', labelpad=15)
        i += 1

    fig.tight_layout()
    savefig(output_path=output_path, savefig_kws=savefig_kws)
