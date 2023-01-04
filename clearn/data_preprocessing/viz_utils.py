"""Utils for visualization"""
# Author: Miguel Alvarez-Garcia

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import seaborn as sns

from ..utils import get_axis, savefig


def missing_values_heatmap(df, output_path=None, savefig_kws=None):
    """
    Plots a heatmap to visualize missing values (light color).

    Parameters
    ----------
    df : `pandas.DataFrame`
       DataFrame containing the data.
    output_path : str, default=None
       Path to save figure as image.
    savefig_kws : dict, default=None
       Save figure options.
    """
    fig, ax = plt.subplots(figsize=(18, 10))
    sns.heatmap(df.isnull().astype(int), cbar=False)
    fig.tight_layout()
    savefig(output_path=output_path, savefig_kws=savefig_kws)


def plot_imputation_pairs_scatter(df, imputation_pairs, sample_frac=1.0, scatter_kws=None, line_kws=None,
                                  output_path=None, savefig_kws=None):
    """
    Plots a grid of scatter plots with every pair of independent-dependent variables used for one-to-one model-based
    imputation.

    Parameters
    ----------
    df : `pandas.DataFrame`
        DataFrame containing the data.
    imputation_pairs : `pandas.DataFrame`
        Imputation pairs as returned by the function `ldbx.data_processing.imputation_pairs()`.
    sample_frac : float, default=1.0
        If < 1 a random sample of every pair of variables will be plotted.
    {scatter,line}_kws : dict, default=None
        Additional keyword arguments to pass to the scatter and line plots.
    output_path : str, default=None
        Path to save figure as image.
    savefig_kws : dict, default=None
        Save figure options.
    """
    scatter_kws = scatter_kws if scatter_kws else dict(color='blue', alpha=0.05)
    line_kws = line_kws if line_kws else dict(color='red')

    ncols = min(4, imputation_pairs.shape[0])
    nrows = int(np.ceil(imputation_pairs.shape[0] / ncols))

    fig, axs = plt.subplots(nrows, ncols, figsize=(4 * ncols, 4 * nrows))
    df_sample = df.sample(frac=sample_frac, random_state=42)
    i = 0
    for idx, row in imputation_pairs.iterrows():
        ax = get_axis(i, axs, ncols, nrows)
        sns.regplot(x=row['var2'], y=row['var1'], data=df_sample, scatter_kws=scatter_kws, line_kws=line_kws, ax=ax)
        ax.set_xlabel(f"{row['var2']} ({'{:.2f}'.format(row['missing_var2'])}% NA)")
        ax.set_ylabel(f"{row['var1']} ({'{:.2f}'.format(row['missing_var1'])}% NA)")
        i += 1

    fig.tight_layout(pad=2)
    savefig(output_path=output_path, savefig_kws=savefig_kws)


def plot_imputation_distribution_assessment(df_prior, df_posterior, imputed_vars, sample_frac=1.0, prior_kws=None,
                                            posterior_kws=None, output_path=None, savefig_kws=None):
    """
    Plots a distribution comparison of each variable with imputed variables, before and after imputation.

    Parameters
    ----------
    df_prior : `pandas.DataFrame`
        DataFrame containing the data before imputation.
    df_posterior : `pandas.DataFrame`
        DataFrame containing the data after imputation.
    imputed_vars : list
        List of variables with imputed variables.
    sample_frac : float, default=1.0
        If < 1 a random sample of every pair of variables will be plotted.
    {prior,posterior}_kws : dict, default=None
        Additional keyword arguments to pass to the kdeplot (https://seaborn.pydata.org/generated/seaborn.kdeplot.html).
    output_path : str, default=None
        Path to save figure as image.
    savefig_kws : dict, default=None
        Save figure options.
    """
    prior_kws = prior_kws if prior_kws else dict(color='#7F3C8D')
    posterior_kws = posterior_kws if posterior_kws else dict(color='#11A579')

    ncols = min(4, len(imputed_vars))
    nrows = int(np.ceil(len(imputed_vars) / ncols))

    fig, axs = plt.subplots(nrows, ncols, figsize=(4 * ncols, 4 * nrows))
    df_prior_sample = df_prior.sample(frac=sample_frac, random_state=42)
    i = 0
    for ivar in imputed_vars:
        ax = get_axis(i, axs, ncols, nrows)
        sns.kdeplot(ivar, data=df_prior_sample, label='Before imputation', ax=ax, **prior_kws)
        sns.kdeplot(ivar, data=df_posterior.loc[df_prior_sample.index], label='After imputation', ax=ax,
                    **posterior_kws)
        ax.legend()
        i += 1

    fig.tight_layout(pad=2)
    savefig(output_path=output_path, savefig_kws=savefig_kws)


def plot_variable_graph_partitioning_components(edges, connected_components, graph_style_kws=None, output_path=None,
                                                savefig_kws=None):
    """
    Plots a connected components of a graph.
    **Note** this function relies on `ldbx.data_processing.variable_graph_partitioning()` for computing the edges and
    connected components.

    Parameters
    ----------
    edges : list of tuples
        List of graph edges
    connected_components : list of sets
        List of connected components.
    graph_style_kws : dict, default=None
        Additional keyword arguments to style graph plots
        (see https://networkx.org/documentation/stable/reference/drawing.html)
    output_path : str, default=None
        Path to save figure as image.
    savefig_kws : dict, default=None
        Save figure options.
    """
    if graph_style_kws is None:
        graph_style_kws = dict(node_size=16, width=0.4, edge_color='grey', node_color='red', with_labels=True)

    nrows = np.shape(connected_components)[0]
    fig, axs = plt.subplots(nrows, 1, figsize=(12, 5*nrows))

    for i in range(nrows):
        fedges = []
        for edge in edges:
            if edge[0] in connected_components[i]:
                fedges.append(edge)

        G = nx.Graph()
        G.add_nodes_from(connected_components[i])
        G.add_edges_from(fedges)

        nx.draw(G, ax=axs[i], **graph_style_kws)
        savefig(output_path=output_path, savefig_kws=savefig_kws)
