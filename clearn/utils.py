# General utils

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pingouin as pg

from matplotlib.ticker import MaxNLocator
from sklearn.metrics import mutual_info_score


def compute_high_corr_pairs(df, corr_thres=0.7, method='pearson'):
    """
    Computes the correlation coefficient between every pair of variables and returns those pairs with an absolute value
    above the given threshold.
    *Note* all variables must be numerical.

    Parameters
    ----------
    df : `pandas.DataFrame`
        DataFrame containing the data.
    corr_thres : float, default=0.7
        Correlation theshold to consider two variables as strongly correlated.
    method : str, default='pearson'
        Method of correlation (pearson, kendall, spearman, or callable function -
        see https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.corr.html).

    Returns
    ----------
    num_pairs : `pandas.DataFrame`
        DataFrame with pairs of highly correlated variables together with the correlation coefficient value.
    """
    corr_df = df.corr(method=method)
    # We only keep the pairs with a corr coefficient above the threshold and melt the DataFrame to have a row per pair
    num_pairs = corr_df.replace({1: 0})[np.abs(corr_df.replace({1: 0})) > corr_thres].reset_index()\
        .melt(id_vars='index', value_vars=corr_df.columns[1:]).dropna().sort_values('value', ascending=False) \
        .rename(columns={'index': 'var1', 'variable': 'var2'}).reset_index(drop=True)
    return num_pairs


def compute_highly_related_mixed_vars(df, num_vars, cat_vars, np2_thres=0.14):
    """
    Computes the dependency between pairs of numerical and categorical variables through partial eta squared, and
    returns those pairs with a value above the given threshold.

    Parameters
    ----------
    df : `pandas.DataFrame`
        DataFrame containing the data.
    num_vars : string, list, series, or vector array
        Numerical variable name(s).
    cat_vars : string, list, series, or vector array
        Categorical variable name(s).
    np2_thres : float, default=0.14
        Threshold to consider two variables as strongly related (see
        https://www.spss-tutorials.com/effect-size/#anova-partial-eta-squared).

    Returns
    ----------
    pairs : `pandas.DataFrame`
        DataFrame with pairs of highly correlated variables together with the partial eta squared value.
    """
    cross_corr_df = cross_corr_ratio(df[cat_vars], df[num_vars])
    # We keep only the pairs with a np2 above the threshold and melt the notebook to have a row per pair
    pairs = cross_corr_df[cross_corr_df > np2_thres].reset_index()\
        .melt(id_vars='index', value_vars=cross_corr_df.columns[1:]) .dropna()\
        .sort_values('value', ascending=False).rename(columns={'index': 'var1', 'variable': 'var2'})\
        .reset_index(drop=True)

    # For every pair <cat_var, num_var>, we're interested in the relationship in both ways
    pairs = pd.concat([pairs, pairs.rename(columns={'var1': 'var2', 'var2': 'var1'})], ignore_index=True)
    pairs = pairs.sort_values(['value', 'var1', 'var2'], ascending=[False, True, True]).reset_index(drop=True)
    return pairs


def compute_highly_related_categorical_vars(df, mi_thres=0.6):
    """
    Computes the dependency between paris of categorical variables through mutual information
    (https://en.wikipedia.org/wiki/Mutual_information), and returns those pairs with a value above the given threshold.

    Parameters
    ----------
    df : `pandas.DataFrame`
        DataFrame containing the data.
    mi_thres : float, default=0.6
        Threshold to consider two variables as strongly related.

    Returns
    ----------
    cat_pairs : `pandas.DataFrame`
        DataFrame with pairs of highly correlated variables together with the mutual information score.
    """
    data = []
    cat_vars = list(df.columns)
    for i in range(len(cat_vars) - 1):
        for j in range(i + 1, len(cat_vars)):
            df_f = df[(~df[cat_vars[i]].isnull()) & (~df[cat_vars[j]].isnull())]
            mis = mutual_info_score(df_f[cat_vars[i]], df_f[cat_vars[j]])
            if mis > mi_thres:
                data.append((cat_vars[i], cat_vars[j], mis))
                data.append((cat_vars[j], cat_vars[i], mis))

    cat_pairs = pd.DataFrame(data=data, columns=['var1', 'var2', 'value']).sort_values(
        ['value', 'var1', 'var2'], ascending=[False, True, True]).reset_index(drop=True)
    return cat_pairs


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
