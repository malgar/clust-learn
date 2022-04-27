# General utils

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from matplotlib.ticker import MaxNLocator


def compute_high_corr_pairs(df, corr_thres=0.8, method='pearson'):
    hi_corr = df.corr(method=method).replace(1, 0)
    hi_corr = hi_corr[np.abs(hi_corr) > corr_thres]
    hi_corr = pd.melt(hi_corr.reset_index(), id_vars='index', value_vars=df.columns, var_name='var2',
                      value_name='corr_coeff').dropna().rename(columns={'index': 'var1'})\
        .sort_values('corr_coeff', ascending=False).reset_index(drop=True)
    return hi_corr


def cross_corr_ratio(df1, df2):
    """
    Calculates the correlation ratio of every column in df1 with every column in df2
    https://en.wikipedia.org/wiki/Correlation_ratio

    Parameters
    ----------
    df1 : `pandas.DataFrame`
    df2 : `pandas.DataFrame`

    Returns
    ----------
    corr_df: `pandas.DataFrame`
        DataFrame with the correlation ratio of every pair of columns from df1 and df2.
    """
    corr_coeffs = []
    df_aux = pd.concat([df1, df2], axis=1)
    for col in df1.columns:
        col_corr_coeffs = []
        for dv in df2.columns:
            col_corr_coeffs.append(pg.anova(data=df_aux, dv=dv, between=col)['np2'].iloc[0])
        corr_coeffs.append(col_corr_coeffs)

    corr_df = pd.DataFrame(corr_coeffs, index=df1.columns, columns=df2.columns).transpose()
    return corr_df


def get_axis(i, axs, ncols, nrows):
    ax = None
    if ncols > 1 and nrows > 1:
        ax = axs[i // ncols, i % ncols]
    elif ncols == 1 and nrows == 1:
        ax = axs
    else:
        ax = axs[i]
    return ax


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
