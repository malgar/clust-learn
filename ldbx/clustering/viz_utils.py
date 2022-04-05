# Utils for visualization

import matplotlib.pyplot as plt
import seaborn as sns

from kneed import KneeLocator
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
