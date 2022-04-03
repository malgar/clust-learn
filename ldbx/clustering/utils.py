# Utils for clustering

import numpy as np

from statsmodels.stats.weightstats import DescrStatsW


def weighted_sum_of_squared_distances(df, cluster_arr, weights=None):
    """
    Calculates the weighted sum of squared distances for given clusters

    Parameters
    ----------
    df : `pandas.DataFrame`
    cluster_arr : `numpy.array`
    weights : `numpy.array`, default=None

    Returns
    ----------
    wssd: float
        Weighted sum of squared distances
    """
    df['cluster'] = cluster_arr
    df['weights'] = 1
    if weights is not None:
        df['weights'] = weights

    wssd = 0
    for cl in df['cluster'].unique():
        df_cl = df[df['cluster'] == cl]
        if df_cl.shape[0] > 1:
            descr = DescrStatsW(df_cl[df_cl.columns], df_cl['weights'])
            wssd += np.sum(np.array(list(map(np.sum, descr.demeaned**2))) * df_cl['weights'])

    return wssd
