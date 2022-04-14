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


def plot_optimal_components_normalized(scores, max_clusters, metric_name, output_path=None, savefig_kws=None):
    fig, ax = plt.subplots(figsize=(8, 5))
    kl = KneeLocator(x=range(1, max_clusters + 1), y=scores, curve='convex', direction='decreasing')
    plot_optimal_normalized_elbow(scores, kl, ax, optimal_label='Optimal number of clusters',
                                  xlabel='Number of clusters', ylabel=f'Normalized {metric_name}')
    savefig(output_path=output_path, savefig_kws=savefig_kws)


def plot_clustercount(df, output_path=None, savefig_kws=None):
    plt.figure(figsize=(df['cluster_cat'].nunique(), 5))
    sns.countplot(x='cluster_cat', data=df, color='#332288', alpha=0.9, order=np.sort(df['cluster_cat'].unique()))
    plt.xticks(rotation=30)
    plt.ylabel('count', fontsize=12, labelpad=10)
    plt.xlabel('clusters', fontsize=12, labelpad=10)
    plt.tight_layout(pad=2)
    savefig(output_path=output_path, savefig_kws=savefig_kws)


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


def plot_clusters_2D(x, y, hue, df, style_kwargs=dict(), output_path=None, savefig_kws=None):

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

    # Left-hand side plot: Scatter plot colored by cluster category
    sns.scatterplot(x, y, hue=hue, data=df.sort_values(hue), alpha=0.3, palette=palette, linewidth=0, ax=axs[0])
    axs[0].vlines(df[x].mean(), 0, 1, color=vline_color, linewidth=1.15, linestyles='--',
                  label=f'Mean {x}')
    axs[0].hlines(df[y].mean(), 0, 1, color=hline_color, linewidth=1.15, linestyles='--',
                  label=f'Mean {y}')
    axs[0].set_xlabel(x, fontsize=12)
    axs[0].set_ylabel(y, fontsize=12)
    axs[0].set_title('Scatter plot by cluster', fontsize=13)
    axs[0].set_xlim(-0.1, 1.1)
    axs[0].set_ylim(-0.05, 1.05)

    # Right-hand side plot: Cluster centroids with optional kernel density area
    sns.scatterplot(x, y, hue=hue, data=df.groupby(hue).mean().reset_index(),
                    alpha=1, palette=palette, linewidth=0, marker='X', s=100, ax=axs[1])

    if kdeplot:
        sns.kdeplot(x=x, y=y, hue=hue, data=df.sort_values(hue), levels=1, alpha=0.2, palette=palette,
                    ax=axs[1])

    axs[1].vlines(df[x].mean(), 0, 1, color=vline_color, linewidth=1, linestyles='--', label=f'Mean {x}')
    axs[1].hlines(df[y].mean(), 0, 1, color=hline_color, linewidth=1, linestyles='--', label=f'Mean {y}')
    axs[1].set_xlabel(x, fontsize=12)
    axs[1].set_ylabel(y, fontsize=12)
    axs[1].set_title('Cluster centroids', fontsize=13)

    axs[0].legend(fontsize=11, title='', title_fontsize=12, labelspacing=0.5,
                  loc=(0.93, 0.5 - 0.167 * (df[hue].nunique() // 4)))
    axs[1].legend(fontsize=11, title='', title_fontsize=12, labelspacing=0.5,
                  loc=(0.93, 0.5 - 0.167 * (df[hue].nunique() // 4)))

    fig.tight_layout(pad=2)
    savefig(output_path=output_path, savefig_kws=savefig_kws)


# TODO: ct contingency table as a DataFrame
def plot_cat_distribution_by_cluster(ct, cat_label=None, cluster_label=None, output_path=None, savefig_kws=None):

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
                color = '#737373' if i < 2 else '#d9d9d9'
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
