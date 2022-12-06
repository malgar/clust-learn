"""Utils for clustering"""
# Author: Miguel Alvarez-Garcia

import numpy as np
import pandas as pd

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
    wssd : float
        Weighted sum of squared distances
    """

    if weights is None:
        weights = np.ones(len(cluster_arr))

    cl_df = pd.DataFrame(data={'cluster': cluster_arr, 'weights': weights})
    cl_df.index = df.index

    wssd = 0
    for cl in cl_df['cluster'].unique():
        if cl_df[cl_df['cluster'] == cl].shape[0] > 1:
            local_weights = cl_df.loc[cl_df['cluster'] == cl, 'weights']
            descr = DescrStatsW(df.loc[cl_df[cl_df['cluster'] == cl].index],
                                local_weights)
            wssd += np.sum(np.array(list(map(np.sum, descr.demeaned ** 2))) * local_weights)

    return wssd
