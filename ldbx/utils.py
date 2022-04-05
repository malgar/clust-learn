# General utils

import matplotlib.pyplot as plt
import numpy as np

from matplotlib.ticker import MaxNLocator


def savefig(output_path=None, savefig_kws=None):
    if output_path is not None:
        if savefig_kws is not None:
            plt.savefig(output_path, **savefig_kws)
        else:
            plt.savefig(output_path, format='jpg', bbox_inches='tight', dpi=300)


def plot_optimal_normalized_elbow(values, kl, ax, optimal_label='', xlabel='', ylabel=''):
    n_components = len(values)

    ax.plot([np.nan] + list(kl.y_normalized), color='#332288', label='')
    ax.plot([np.nan] + list(kl.y_difference), color='#008695', label='Difference curve')

    ax.axvline(kl.knee, linestyle='--', linewidth=1, color='#E73F74', label=optimal_label)
    ax.set_xticks(np.append(ax.get_xticks()[1:-1], [kl.knee]))
    ax.set_xlim(-n_components * 0.02, n_components * 1.02)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    ax.set_xlabel(xlabel, fontsize=13, labelpad=15)
    ax.set_ylabel(ylabel, fontsize=13, labelpad=8)
    ax.legend(fontsize=12, labelspacing=0.5)
