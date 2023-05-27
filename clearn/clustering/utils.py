"""Utils for clustering"""
# Author: Miguel Alvarez-Garcia

import inspect
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


def weighted_mean(data, weights=None):
    """
    Calculates the weighted mean of an array/list.

    Parameters
    ----------
    data : `numpy.array` or list, default=None
         Array containing data to be averaged.
    weights : `numpy.array` or list
       An array of weights associated with the values in `data`

    Returns
    ----------
    res : float
        (Weighted) mean.
    """
    return np.average(data, weights=weights)


def weighted_std(data, weights=None):
    """
    Calculates the weighted standard deviation of an array/list.

    Parameters
    ----------
    data : `numpy.array` or list, default=None
         Array containing data.
    weights : `numpy.array` or list
       An array of weights associated with the values in `data`

    Returns
    ----------
    res : float
        (Weighted) standard deviation.
    """
    average = np.average(data, weights=weights)
    variance = np.average((data-average)**2, weights=weights)
    return np.sqrt(variance)


def is_sklearn_compatible(algorithm):
    return 'fit_predict' in dir(algorithm) and 'set_params' in dir(algorithm)


def accepts_param(function, param):
    args, varargs, varkw, defaults, kwonlyargs, kwonlydefaults, annotations = inspect.getfullargspec(function)
    return param in args


def get_n_clusters_param_name(algorithm):
    param_list = list(algorithm.get_params().keys())
    if 'n_clusters' in param_list:
        return 'n_clusters'
    elif 'n_components' in param_list:
        return 'n_components'
    else:
        return None
