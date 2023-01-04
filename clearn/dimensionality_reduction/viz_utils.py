"""Visualization utils for dimensionality reduction"""
# Author: Miguel Alvarez-Garcia

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from kneed import KneeLocator
from matplotlib.gridspec import GridSpec

from .table_utils import cat_main_contributors, num_main_contributors
from ..utils import get_axis, plot_optimal_normalized_elbow, savefig

sns.set_style('whitegrid')

__types__ = ['cumulative', 'ratio', 'normalized']


def _plot_cumulative_explained_var(explained_variance_ratio, kl, thres, ax):
    n_components = len(explained_variance_ratio)

    ax.plot(np.append([0], explained_variance_ratio.cumsum()), label='', color='#332288')

    ax.axhline(thres, 0.01, 0.99, linestyle='--', linewidth=1, color='grey',
               label=f'{int(thres * 100)}% Explained variance')
    ax.set_yticks(np.append(ax.get_yticks()[1:-1], [thres]))

    ax.axvline(kl.knee, linestyle='--', linewidth=1, color='#E73F74', label=f'Optimal number of components')
    ax.axvline((explained_variance_ratio.cumsum() < thres).sum()+1, linestyle='--', linewidth=1, color='#11A579',
               label=f'Minimum number of components for {int(thres * 100)}% explained variance')
    ax.set_xticks(np.append(ax.get_xticks()[1:-1], [kl.knee, (explained_variance_ratio.cumsum() < thres).sum()+1]))
    ax.set_xlim(-n_components * 0.02, n_components * 1.02)

    ax.set_ylabel('Explained variance (cumulative ratio)', fontsize=13, labelpad=15)
    ax.legend(fontsize=12, labelspacing=0.5)


def _plot_explained_var_ratio(explained_variance_ratio, kl, ax):
    n_components = len(explained_variance_ratio)

    ax.plot([np.nan] + list(explained_variance_ratio), color='#332288', label='')

    avg_explained_var = 1 / (n_components - 1)
    ax.axhline(avg_explained_var, 0.01, 0.99, linestyle='--', linewidth=1, color='grey',
               label='Average explained variance (%)')
    ax.set_yticks(np.append(ax.get_yticks()[1:-1], [avg_explained_var]))

    ax.axvline(kl.knee, linestyle='--', linewidth=1, color='#E73F74', label=f'Optimal number of components')
    ax.axvline((explained_variance_ratio > avg_explained_var).sum(), linestyle='--', linewidth=1,
               color='#11A579', label=f'Number of components above average explained variance')
    ax.set_xticks(
        np.append(ax.get_xticks()[1:-1], [kl.knee, (explained_variance_ratio > avg_explained_var).sum()]))
    ax.set_xlim(-n_components * 0.02, n_components * 1.02)

    ax.set_ylabel('Explained variance (ratio)', fontsize=13, labelpad=8)
    ax.legend(fontsize=12, labelspacing=0.5)


def _plot_normalized_explained_var(explained_variance_ratio, kl, ax):
    plot_optimal_normalized_elbow(explained_variance_ratio, kl, ax, optimal_label='Optimal number of components',
                                  xlabel='Number of components', ylabel='Normalized explained variance curve')


def plot_explained_variance(explained_variance_ratio, thres=0.5, plots='all', output_path=None, savefig_kws=None):
    """
    Plot the explained variance (ratio, cumulative, and/or normalized)

    Parameters
    ----------
    explained_variance_ratio : `numpy.array`
        Array with the explained variance ratio of the first n components.
        Note it is assumed this is provided in descending order.
    thres : float, default=0.5
        Minimum explained cumulative variance ratio.
    plots : str or list, default='all'
        The following plots are supported: ['cumulative', 'ratio', 'normalized']
    output_path : str, default=None
        Path to save figure as image.
    savefig_kws : dict, default=None
        Save figure options.
    """

    n_components = len(explained_variance_ratio)

    if plots == 'all':
        plots = __types__
    if not isinstance(plots, list):
        plots = [plots]

    kl = KneeLocator(x=range(1, n_components + 1), y=explained_variance_ratio, curve='convex',
                     direction='decreasing')

    fig, axs = plt.subplots(len(plots), 1, figsize=(8, 5 * len(plots)))

    i = 0
    for p in plots:

        ax = axs
        if len(plots) > 1:
            ax = axs[i]

        if p == 'cumulative':
            _plot_cumulative_explained_var(explained_variance_ratio, kl, thres, ax)
        elif p == 'ratio':
            _plot_explained_var_ratio(explained_variance_ratio, kl, ax)
        elif p == 'normalized':
            _plot_normalized_explained_var(explained_variance_ratio, kl, ax)
        else:
            raise NameError(f"Plot type '{p}' does not exists")
        i += 1

    plt.tight_layout(pad=2)
    savefig(output_path=output_path, savefig_kws=savefig_kws)


def plot_num_main_contributors(df, df_trans, thres=0.5, n_contributors=5, dim_idx=None, output_path=None,
                               savefig_kws=None):
    """
    Plot main contributors (original variables with the strongest relation with derived variables) for
    every derived variable

    Parameters
    ----------
    df : `pandas.DataFrame`
        DataFrame with original numerical variables.
    df_trans : `pandas.DataFrame`
        DataFrame with derived variables.
    thres : float, default=0.5
        Minimum Pearson correlation coefficient to consider an original and a derived variable to be strongly related.
    n_contributors : int, default=5
        Number of contributors by derived variables (the ones with the strongest correlation coefficient
        are shown).
    dim_idx : int, default=None
        In case only main contributors for derived variable in column position dim_idx are retrieved (starts at 0).
    output_path : str, default=None
        Path to save figure as image.
    savefig_kws : dict, default=None
        Save figure options.
    """

    cmap = matplotlib.cm.get_cmap('coolwarm')
    n_contributors = np.minimum(n_contributors, df.shape[1])
    mc = num_main_contributors(df, df_trans, thres=thres, n_contributors=n_contributors, dim_idx=dim_idx)
    mc['corr_coeff_abs'] = np.abs(mc['corr_coeff'])
    mc = mc.sort_values(by=['component', 'corr_coeff_abs']).reset_index(drop=True).drop(columns='corr_coeff_abs')
    nplots = mc['component'].nunique()
    ncols = 2
    if nplots == 1:
        ncols = 1
    elif nplots % 2 > 0 or nplots % 3 == 0:
        ncols = 3

    nbars = mc.groupby('component')['var_name'].count().max()

    nrows = nplots // ncols + (nplots % ncols > 0)
    fig, axs = plt.subplots(nrows, ncols, figsize=(6 * ncols, 0.6 * nbars * nrows))
    xticks = (np.array(range(9)) - 4) / 4

    i = 0
    for pc in mc['component'].unique():
        ax = get_axis(i, axs, ncols, nrows)
        n_pc_contrib = mc[mc['component'] == pc].shape[0]
        ax.barh(y=range(n_pc_contrib), width=mc.loc[mc['component'] == pc, 'corr_coeff'],
                color=list(map(lambda x: cmap(x), (mc.loc[mc['component'] == pc, 'corr_coeff'] + 1) / 2)), alpha=0.95)
        ax.vlines(0, -0.5, n_pc_contrib - 0.5, color='black', linewidth=0.5)
        if i // ncols == nrows - 1:
            ax.set_xlabel('Correlation coefficient', fontsize=12)
        ax.set_xticks(ticks=xticks)
        ax.set_yticks(ticks=range(n_pc_contrib))
        ax.set_yticklabels(labels=mc.loc[mc['component'] == pc, 'var_name'], rotation=0, fontsize=11)
        ax.set_title(str.upper(pc), fontsize=13)
        i += 1

    while i < ncols*nrows:
        ax = get_axis(i, axs, ncols, nrows)
        ax.axis('off')
        i += 1

    fig.tight_layout(pad=2)
    savefig(output_path=output_path, savefig_kws=savefig_kws)


def plot_cat_main_contributor_distribution(df, df_trans, thres=0.14, n_contributors=None, dim_idx=None,
                                           output_path=None, savefig_kws=None):
    """
    Plot main contributors (original variables with the strongest relation with derived variables) for
    every derived variable

    Parameters
    ----------
    df : `pandas.DataFrame`
        DataFrame with original categorical variables.
    df_trans : `pandas.DataFrame`
        DataFrame with derived variables.
    thres : float, default=0.5
         Minimum correlation ratio to consider an original and a derived variable to be strongly related.
    n_contributors: int, default=5
        Number of contributors by derived variables (the ones with the strongest correlation coefficient
        are shown).
    dim_idx : int, default=None
        In case only main contributors for derived variable in column position dim_idx are retrieved (starts at 0).
    output_path : str, default=None
        Path to save figure as image.
    savefig_kws : dict, default=None
        Save figure options.
    """
    dim_name = ''
    if dim_idx is not None:
        dim_name = df_trans.columns[dim_idx]
        df_trans = df_trans[dim_name].to_frame()
    else:
        raise RuntimeWarning(
            '''`plot_disc_main_contributor_distribution` is designed to plot one component at a time. 
            Provide a value for dim_idx''')

    mc = cat_main_contributors(df, df_trans, thres=thres, n_contributors=n_contributors, dim_idx=dim_idx)

    ncols = mc.shape[0]
    if ncols > 5:
        raise Warning(f'''{ncols} original variables are highly related to de new construct variable.
                      Only the strongest 5 will be shown.''')
        ncols = 5

    nrows = np.lcm.reduce(df[mc['var_name'].tolist()].nunique().tolist())

    fig = plt.figure(figsize=(np.minimum(6 * ncols, 20), 5))
    gs = GridSpec(nrows, ncols, figure=fig)
    df = pd.concat([df, df_trans], axis=1)

    j = 0
    for idx, row in mc.iterrows():
        var_name = row['var_name']
        nvalues = df[var_name].nunique()

        i = 0
        ax0 = None
        for v in df[var_name].unique():
            ax = fig.add_subplot(gs[i:i + int(nrows / nvalues), j], sharex=ax0, sharey=ax0)
            sns.kdeplot(data=df[df[var_name] == v], x=dim_name, color='blue', fill=True, ax=ax)
            ax.set_title(f'{var_name} = {v}', fontsize=10)
            ax.set_ylabel('')

            i += int(nrows / nvalues)
            if i < nrows:
                ax.set_xlabel('')

            if ax0 is None:
                ax0 = ax
        j += 1

    sns.despine(fig)
    fig.supylabel('Density functions', fontsize=13, x=0.01)
    fig.tight_layout(w_pad=3)
    savefig(output_path=output_path, savefig_kws=savefig_kws)


def plot_cumulative_explained_var_comparison(explained_variance_ratio1, explained_variance_ratio2, name1=None,
                                             name2=None, thres=None, output_path=None, savefig_kws=None):
    """
    Plots comparison of cumulative explained variance between two techniques.

    Parameters
    ----------
    explained_variance_ratio1 : list, `numpy.array`
        Explained variance ratio by technique 1.
    explained_variance_ratio2 : list, `numpy.array`
        Explained variance ratio by technique 2.
    name1 : str, default=None
        Name of technique 1. (For styling purposes).
    name2 : str, default=None
        Name of technique 2. (For styling purposes).
    thres : float, default=None
         Reference threshold for cumulative explained variance ratio. (For styling purposes).
    output_path : str, default=None
        Path to save figure as image.
    savefig_kws : dict, default=None
        Save figure options.
    """
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(np.append([0], explained_variance_ratio1), color='#7F3C8D', label=name1)
    ax.plot(np.append([0], explained_variance_ratio2), color='#11A579', label=name2)
    if thres is not None:
        ax.axhline(thres, 0.01, 0.99, linestyle='--', linewidth=1, color='grey',
                   label=f'{int(thres*100)}% Explained variance')
        ax.set_yticks(np.append(ax.get_yticks()[1:-1], [thres]))
    n_components = np.maximum(len(explained_variance_ratio1), len(explained_variance_ratio2))
    ax.set_xlim(-n_components * 0.02, n_components * 1.02)
    ax.set_ylabel('Explained variance (cumulative ratio)', fontsize=13, labelpad=15)
    ax.legend(fontsize=12, labelspacing=0.5)
    ax.set_xlabel('Number of components', fontsize=13, labelpad=15)
    ax.legend(fontsize=12, labelspacing=0.5)
    fig.tight_layout(w_pad=2)
    savefig(output_path=output_path, savefig_kws=savefig_kws)


def plot_compare_pca_based_components(components_pca, components_other, original_vars, other_name='Sparse PCA', n_pc=1,
                                      output_path=None, savefig_kws=None):
    """
    Plots comparison of cumulative explained variance between two techniques.

    Parameters
    ----------
    components_pca : `numpy.array`
        Components of the `n_pc` principal component calculated with PCA.
    components_other : `numpy.array`
        Components of the `n_pc` principal component calculated a PCA based technique.
    original_vars : list
        List of names of the original variables.
    other_name : str, default='Sparse PCA'
        Name of the other applied technique.
    n_pc : int, default=1
        Principal component to which the components refer to.
    output_path : str, default=None
        Path to save figure as image.
    savefig_kws : dict, default=None
        Save figure options.
    """
    plt.figure(figsize=(16, 5))
    plt.bar(x=range(len(components_pca)), height=components_pca, label='PCA', color='#11A579', alpha=0.5)
    plt.plot(range(len(components_other)), components_other, 'o', label=other_name, color='#CF1C90', alpha=0.67)

    plt.hlines(0, 0, len(components_other), color='black', linewidth=0.5)
    plt.ylabel('Coefficients', fontsize=12)
    plt.xticks(ticks=range(len(original_vars)), labels=original_vars, rotation=90, fontsize=11)
    plt.legend(fontsize=12, title='Method', title_fontsize=13, labelspacing=0.5)
    plt.xlim(-1, len(original_vars))
    plt.title(f'Principal Component {str(n_pc).zfill(2)}', fontsize=14)
    plt.tight_layout()
    savefig(output_path=output_path, savefig_kws=savefig_kws)
