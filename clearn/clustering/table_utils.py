"""Table statistics utils for clustering"""
# Author: Miguel Alvarez-Garcia

from .utils import *


def compare_cluster_means_to_global_means(df, dimensions, weights=None, data_standardized=False, output_path=None):
    """
    For every cluster and every variable in `dimensions`, the relative difference between the intra-cluster mean
    and the global mean is computed.

    Parameters
    ----------
    df : `pandas.DataFrame`
        DataFrame with dimension columns.
    dimensions: list or `np.array`
        List of variables to compare.
    weights: `np.array`, default=None
        Sample weights.
    data_standardized: bool, default=False
        Indicates whether data in `df[dimensions]` is standardized (mean=0, std=1)
    output_path : str, default=None
        If an output_path is passed, the resulting DataFame is saved as a CSV file.

    Returns
    ----------
    df_agg_diff : `pandas.DataFrame`
        DataFrame with the comparison.
    """
    agg_method = 'mean'
    if weights is not None:
        def wmean(x): return weighted_mean(x, weights[x.index])
        agg_method = wmean
    df_agg = df.groupby('cluster').agg(dict(zip(list(dimensions), [agg_method] * len(dimensions))))

    df_agg_diff = df_agg.copy()
    if data_standardized:
        df_agg_diff = df_agg
    else:
        mean_array = df[dimensions].apply(agg_method).values
        for idx, row in df_agg.iterrows():
            df_agg_diff.loc[idx, dimensions] = (row[dimensions] - mean_array) / mean_array

    df_agg_diff = df_agg_diff.reset_index()
    if output_path is not None:
        df_agg_diff.to_csv(output_path, index=False)

    return df_agg_diff
