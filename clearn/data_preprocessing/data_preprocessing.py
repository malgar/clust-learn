"""Data preprocessing"""
# Author: Miguel Alvarez-Garcia

import networkx as nx
import numpy as np
import pandas as pd

from scipy.stats import rv_discrete
from sklearn.ensemble import IsolationForest
from sklearn.feature_selection import mutual_info_regression, mutual_info_classif
from sklearn.impute import KNNImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mutual_info_score
from sklearn.preprocessing import MinMaxScaler
from ..utils import compute_high_corr_pairs, compute_highly_related_categorical_vars, compute_highly_related_mixed_vars


def compute_missing(df, normalize=True):
    """
    Calculates the pct/count of missing values per column.

    Parameters
    ----------
    df : `pandas.DataFrame`
    normalize : boolean, default=True

    Returns
    ----------
    missing_df : `pandas.DataFrame`
        DataFrame with the pct/counts of missing values per column.
    """
    missing_df = (df.isnull().sum()).to_frame('missing').reset_index().rename(columns={'index': 'var_name'})
    if normalize:
        missing_df['missing'] = missing_df['missing'] * 100 / df.shape[0]
    missing_df = missing_df.sort_values('missing', ascending=False)
    return missing_df


def num_imputation_pairs(df, corr_thres=0.7, method='pearson'):
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
    return compute_high_corr_pairs(df, corr_thres=corr_thres, method=method)


def mixed_imputation_pairs(df, num_vars, cat_vars, np2_thres=0.14):
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
    return compute_highly_related_mixed_vars(df, num_vars, cat_vars, np2_thres)


def cat_imputation_pairs(df, mi_thres=0.6):
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
    return compute_highly_related_categorical_vars(df, mi_thres)


def imputation_pairs(df, num_vars=None, cat_vars=None, num_kws=None, mixed_kws=None, cat_kws=None):
    """
    Computes one-to-one model-based imputation pairs. These are strongly related pairs of variables. Depending on the
    type of variables, a correlation coefficient (numerical variables), partial eta squared (mixed-type variables), or
    mutual information (categorical variables) is used to measured the strength of the relationship of each pair.

    When one variable is strongly related to more than one variable, the one that allows a larger amount of imputations.

    Parameters
    ----------
    df : `pandas.DataFrame`
        DataFrame containing the data.
    num_vars : str, list, series, or vector array
        Numerical variable name(s).
    cat_vars : str, list, series, or vector array
        Categorical variable name(s).
    {num,mixed,cat}_kws : dict, default=None
        Additional keyword arguments to pass to `num_imputation_pairs()`, `mixed_imputation_pairs()`, and
        `cat_imputation_pairs()`.

    Returns
    ----------
    final_pairs : `pandas.DataFrame`
        DataFrame with pairs of highly correlated variables (var1: variable with values to impute; var2: variable to be
        used as independent variable for model-based imputation), together proportion of missing values of variables
        var1 and var2.
    """
    if num_vars is None and cat_vars is None:
        raise ValueError('Numerical or categorical variable lists are required.')

    # Numerical variable pairs (correlation)
    num_pairs = pd.DataFrame()
    if num_vars:
        num_kws = num_kws if num_kws else dict()
        num_pairs = num_imputation_pairs(df[num_vars], **num_kws)

    # Mixed variable pairs (cross correlation)
    mixed_pairs = pd.DataFrame()
    if num_vars and cat_vars:
        mixed_kws = mixed_kws if mixed_kws else dict()
        mixed_pairs = mixed_imputation_pairs(df, num_vars, cat_vars, **mixed_kws)

    # Categorical variable pairs (mutual information)
    cat_pairs = pd.DataFrame()
    if cat_vars:
        cat_kws = cat_kws if cat_kws else dict()
        cat_pairs = cat_imputation_pairs(df[cat_vars], **cat_kws)

    final_pairs = pd.concat([num_pairs, mixed_pairs, cat_pairs], ignore_index=True)
    final_pairs = final_pairs.drop(columns='value')

    num_vars = num_vars if num_vars else []
    cat_vars = cat_vars if cat_vars else []

    # We only want to keep those pairs with missing values
    missing_df = compute_missing(df[num_vars + cat_vars])
    final_pairs = final_pairs.merge(missing_df, left_on='var1', right_on='var_name').rename(
        columns={'missing': 'missing_var1'}).drop(columns='var_name')
    final_pairs = final_pairs.merge(missing_df, left_on='var2', right_on='var_name').rename(
        columns={'missing': 'missing_var2'}).drop(columns='var_name')
    # Each variable with missing values and strongly related to at least another one, is assigned to the one from which
    # a larger proportion of missing values can be imputed.
    final_pairs = final_pairs.loc[final_pairs.groupby('var1')['missing_var2'].idxmin()].sort_values(
        ['missing_var2', 'missing_var1'])
    final_pairs = final_pairs[final_pairs['missing_var1'] > final_pairs['missing_var2']].reset_index(drop=True)
    return final_pairs


def _bucketize(values, q=10):
    """
    Quantile bucketizer.

    Parameters
    ----------
    values : `numpy.array`, list
        Array of values to bucketize.
    q : int, default=10
        Number of buckets.

    Returns
    ----------
    b_values : `pandas.Series`
        Series with bucketized values.
    """
    b_values = values
    if len(np.unique(values)) > 10:
        b_values = pd.qcut(values, q=q, duplicates='drop')
    return b_values


def impute_missing_values_with_highly_related_pairs(df_original, imputation_pairs, num_vars=None, cat_vars=None):
    """
    One-to-one model based imputation for the imputation pairs identified.
    * If both variables are numerical, linear regression is used. The regressor is trained using all observations with
    known values for both variables.
    * If both variables are categorical, we use hot deck imputation considering for every recipient all observations
    with the same value of the independent variable as donors. The value to be imputed is selected at random following
    the discrete empirical distribution.
    * For mixed-type variables, the numerical one is discretized by bucketization into quantiles and both variables are
    treated as categorical.

    Parameters
    ----------
    df_original : `pandas.DataFrame`
        DataFrame containing the data.
    imputation_pairs : `pandas.DataFrame`
        Imputation pairs as returned by the function `imputation_pairs()`.
    num_vars : str, list, series, or vector array
        Numerical variable name(s).
    cat_vars : str, list, series, or vector array
        Categorical variable name(s).

    Returns
    ----------
    df : `pandas.DataFrame`
        DataFrame with imputed data.
    """
    if num_vars is None and cat_vars is None:
        raise ValueError('Numerical and categorical variable lists are required.')

    num_vars = num_vars if num_vars else []
    cat_vars = cat_vars if cat_vars else []
    df = df_original.copy()

    for idx, row in imputation_pairs.iterrows():
        if row['var1'] in num_vars and row['var2'] in num_vars:
            # If both variables are numerical, use linear regression
            print(f"Imputing with linear regression {row['var2']} -> {row['var1']}")
            # First, model is fit for non-missing values
            lr = LinearRegression()
            df_f = df[(~df[row['var2']].isnull()) & (~df[row['var1']].isnull())]
            lr.fit(df_f[row['var2']].values.reshape(-1, 1), df_f[row['var1']])
            # Secondly, the fitted model is used for imputation
            df_f = df[(~df[row['var2']].isnull()) & (df[row['var1']].isnull())]
            df.loc[df_f.index, row['var1']] = lr.predict(df_f[row['var2']].values.reshape(-1, 1))

        else:
            print(f"Imputing with empirical discrete distribution {row['var2']} -> {row['var1']}")
            # Bucketize the numerical variable to treat both as categorical.
            if row['var2'] in num_vars:
                # print(f"Bucketize variable {row['var2']}")
                df[f"b_{row['var2']}"] = _bucketize(df[row['var2']])
                df[f"b_{row['var1']}"] = df[row['var1']]
            elif row['var1'] in num_vars:
                # print(f"Bucketize variable {row['var1']}")
                df[f"b_{row['var1']}"] = _bucketize(df[row['var1']])
                df[f"b_{row['var2']}"] = df[row['var2']]
            else:
                df[f"b_{row['var1']}"] = df[row['var1']]
                df[f"b_{row['var2']}"] = df[row['var2']]

            # Impute using the empirical discrete distribution function
            df_f = df[(~df[row['var2']].isnull()) & (df[row['var1']].isnull())]
            predictor_values = df_f[f"b_{row['var2']}"].unique().tolist()
            for pv in predictor_values:
                # For every value of the independent variable, we compute the discrete distribution frequencies of the
                # dependent variable calculated when both dependent and independent values are non-missing.
                dist_freq = df.loc[(df[f"b_{row['var2']}"] == pv) & (~df[row['var1']].isnull()), row['var1']]\
                    .value_counts(normalize=True).sort_index()
                if dist_freq.shape[0] > 0:
                    xk = tuple(dist_freq.index)
                    pk = tuple(dist_freq)
                    disc_dist = rv_discrete(name='imputation', values=(xk, pk))
                    # Impute missing values for the dependent variable using a random value extracted from the empirical
                    # discrete distribution computed above
                    no_missing = df[(df[f"b_{row['var2']}"] == pv) & (df[row['var1']].isnull())].shape[0]
                    df.loc[(df[f"b_{row['var2']}"] == pv) & (df[row['var1']].isnull()), row['var1']] = disc_dist.ppf(
                        np.random.rand(no_missing))

            df = df.drop(columns=[f"b_{row['var1']}", f"b_{row['var2']}"])

    return df


def mutual_information_pair_scores(df, num_vars=None, cat_vars=None):
    """
    Computes the mutual information score for every pair of variables.

    Parameters
    ----------
    df : `pandas.DataFrame`
        DataFrame containing the data.
    num_vars : str, list, series, or vector array
        Numerical variable name(s).
    cat_vars : str, list, series, or vector array
        Categorical variable name(s).

    Returns
    ----------
    mi_scores_df : `pandas.DataFrame`
        DataFrame with mutual information score for every pair of variables.
    """
    if num_vars is None and cat_vars is None:
        raise ValueError('Numerical and categorical variable lists are required.')

    num_vars = num_vars if num_vars else []
    cat_vars = cat_vars if cat_vars else []

    # First, we calculate max mutual information per variable for normalization
    max_mi_scores = dict()
    for nv in num_vars:
        values = df[nv].dropna().values
        mi = mutual_info_regression(values.reshape(-1, 1), values, discrete_features=False, random_state=42)[0]
        max_mi_scores[nv] = mi
    for cv in cat_vars:
        values = df[cv].dropna().values
        mi = mutual_info_score(values, values)
        max_mi_scores[cv] = mi

    # Secondly, we compute the mutual information of every pair of variables and normalize it by the max mutual
    # information calculated above
    mi_scores = list()
    # Numerical pairs
    for i in range(len(num_vars) - 1):
        for j in range(i + 1, len(num_vars)):
            df_f = df[[num_vars[i], num_vars[j]]].dropna()
            mi = max(mutual_info_regression(df_f[num_vars[i]].values.reshape(-1, 1), df_f[num_vars[j]],
                                            discrete_features=False, random_state=42)[0],
                     mutual_info_regression(df_f[num_vars[j]].values.reshape(-1, 1), df_f[num_vars[i]],
                                            discrete_features=False, random_state=42)[0])
            norm_factor = max(max_mi_scores[num_vars[i]], max_mi_scores[num_vars[j]])
            mi = mi / norm_factor
            mi_scores.append((num_vars[i], num_vars[j], mi))

    # Mixed pairs
    for i in range(len(num_vars)):
        for j in range(len(cat_vars)):
            df_f = df[[num_vars[i], cat_vars[j]]].dropna()
            mi = max(mutual_info_classif(df_f[num_vars[i]].values.reshape(-1, 1), df_f[cat_vars[j]],
                                         discrete_features=False, random_state=42)[0],
                     mutual_info_regression(df_f[cat_vars[j]].values.reshape(-1, 1), df_f[num_vars[i]],
                                            discrete_features=True, random_state=42)[0])
            norm_factor = max(max_mi_scores[num_vars[i]], max_mi_scores[cat_vars[j]])
            mi = mi / norm_factor
            mi_scores.append((num_vars[i], cat_vars[j], mi))

    # Categorical pairs
    for i in range(len(cat_vars) - 1):
        for j in range(i + 1, len(cat_vars)):
            df_f = df[[cat_vars[i], cat_vars[j]]].dropna()
            mi = mutual_info_score(df_f[cat_vars[i]], df_f[cat_vars[j]])
            norm_factor = max(max_mi_scores[cat_vars[i]], max_mi_scores[cat_vars[j]])
            mi = mi / norm_factor
            mi_scores.append((cat_vars[i], cat_vars[j], mi))

    mi_scores_df = pd.DataFrame(mi_scores, columns=['var1', 'var2', 'score'])
    return mi_scores_df


def variable_graph_partitioning(pair_scores, thres=0.05):
    """
    Computes a graph partitioning of the graph G=(V, E), where V = {variables} and E = {pairs of variables with a mutual
    information score above the threshold}. The connected components represent clusters of related variables.

    Parameters
    ----------
    pair_scores : `pandas.DataFrame`
        DataFrame with mutual information score for every pair of variables.
    thres : float, default=0.05
        Threshold to determine if two variables are similar based on mutual information score.

    Returns
    ----------
    all_nodes : list
        DataFrame with mutual information score for every pair of variables.
    edges : list of tuples
        List of graph edges
    g_list : list of sets
        List of connected components.
    """
    edges = pair_scores[pair_scores['score'] > thres].apply(lambda row: (row['var1'], row['var2']), axis=1).tolist()

    connected_nodes = list(zip(*edges))
    connected_nodes = list(connected_nodes[0]) + list(connected_nodes[1])
    nodes = list(np.unique(connected_nodes))

    G = nx.Graph()
    G.add_nodes_from(nodes)
    G.add_edges_from(edges)
    g_list = [c for c in sorted(nx.connected_components(G), key=len, reverse=True)]
    # Nodes, edges, connected components
    all_nodes = list(np.unique(pair_scores['var1'].to_list() + pair_scores['var2'].to_list()))
    return all_nodes, edges, g_list


def _hot_deck_weights(distances):
    """
    Callable function that selects a random neighbor for every missing value. This function accepts an array of
    distances, and returns an array of the same shape containing 0-1 weights.

    Parameters
    ----------
    distances : `numpy.ndarray`
        Array of distances of a recipient with its KNN neighbors.

    Returns
    ----------
    weights : `numpy.ndarray`
        Array with 0-1 weights.
    """
    weights = np.zeros(np.shape(distances))

    # Randomly select the neighbor that will be used to impute each of the missing values
    ids = np.random.randint(0, np.shape(distances)[1], size=np.shape(distances)[0])
    if np.shape(distances)[0] == 1:
        ids = [ids]

    i = 0
    for idx in ids:
        weights[i][idx] = 1
        i += 1

    return weights


def hot_deck_imputation(df_original, variables, k=8, partitions=None):
    """
    Computes KNN hot deck imputation using sklearn KNNImputer
    (see https://scikit-learn.org/stable/modules/generated/sklearn.impute.KNNImputer.html).

    Parameters
    ----------
    df_original : `pandas.DataFrame`
        DataFrame containing the data.
    variables : list
        List of variables with potential missing values.
        **Note** all partitions must be contained in this variable list.
    k : int, default=8
        Number of neighbors.
    partitions : `numpy.ndarray`, list of lists
        Variable partitions/clusters to impute variables by cluster.
        **Note** if any partition is passed, imputation is only performed on those observations with fewer than 1/4 of
        their values within the partition variables missed.

    Returns
    ----------
    df : `pandas.DataFrame`
        DataFrame with imputed values.
    """
    apply_threshold = True
    if partitions is None:
        apply_threshold = False
        partitions = [set(variables)]

    original_index = df_original.index
    df = df_original.sample(frac=1, random_state=42)

    mms = MinMaxScaler()
    df[variables] = mms.fit_transform(df[variables])

    for p in partitions:
        p = sorted(p)
        print('Partition:', p)
        imputer = KNNImputer(n_neighbors=k, weights=_hot_deck_weights)
        df_p = df
        if apply_threshold:
            df_p = df[df[p].isnull().sum(1) <= np.floor(len(p) / 4)]
        df_p_i = pd.DataFrame(data=imputer.fit_transform(df_p[p]), index=df_p.index, columns=p)
        df.loc[df_p_i.index, df_p_i.columns] = df_p_i.values
        print('Remaining missing values', df.isnull().sum().sum())

    df[variables] = mms.inverse_transform(df[variables])
    df = df.loc[original_index]
    return df


def impute_missing_values(df, num_vars, cat_vars, num_pair_kws=None, mixed_pair_kws=None, cat_pair_kws=None,
                          graph_thres=0.05, k=8, max_missing_thres=0.33):
    """
    Main function for data imputation. The procedure consists of four different steps:
     1. One-to-one model based imputation for strongly related variables (see
        `impute_missing_values_with_highly_related_pairs` doc).
     2. Cluster based hot deck imputation. Clusters of strongly related variables are computed using mutual information
        between pairs of variables and calculating the connected components of the undirected graph G=(V, E), where V is
        the set of variables and E the pairs of variables with mutual information above a predefined threshold.
     3. Records with a proportion of missing values above a predefined threshold are discarded to ensure the quality of
        the hot deck imputation.
     4. Hot deck imputation for the remaining missing values considering all variables together.

    Parameters
    ----------
    df : `pandas.DataFrame`
        DataFrame containing the data.
    num_vars : str, list, series, or vector array
        Numerical variable name(s).
    cat_vars : str, list, series, or vector array
        Categorical variable name(s).
    {num,mixed,cat}_pair_kws : dict, default=None
        Additional keyword arguments to pass to compute imputation_pairs.
    graph_thres : float, default=0.05
        Threshold to determine if two variables are similar based on mutual information score, and therefore should be
        considered as edges of the graph from which variable clusters are derived.
    k : int, default=8
        Number of neighbors to consider in hot deck imputation.
    max_missing_thres: float, default=0.33
        Max proportion of missing values per record allowed before final general hot deck imputation.

    Returns
    ----------
    df : `pandas.DataFrame`
        DataFrame with all missing values imputed.
    """
    # One-to-one model based imputation for strongly related variables
    print('--- One-to-one model based imputation for strongly related variables ---')
    ip = imputation_pairs(df, num_vars, cat_vars, num_kws=num_pair_kws, mixed_kws=mixed_pair_kws, cat_kws=cat_pair_kws)
    df = impute_missing_values_with_highly_related_pairs(df, ip, num_vars, cat_vars)

    # Cluster based hot deck imputation
    # Applied only to clusters with at least 4 variables
    print('--- Cluster based hot deck imputation ---')
    mi_scores = mutual_information_pair_scores(df, num_vars, cat_vars)
    nodes, edges, comp = variable_graph_partitioning(mi_scores, thres=graph_thres)
    df = hot_deck_imputation(df, num_vars + cat_vars, k=k,
                             partitions=list(np.array(comp)[np.array(list(map(len, comp))) >= 4]))

    # Remove records with more than max_missing_thres missing values are removed.
    print(f'--- Removing records with more than {max_missing_thres*100} missing values ---')
    df = df[df.isnull().sum(1) < df.shape[1] * max_missing_thres]

    # Hot deck imputation for the remaining missing values
    print('--- Hot deck imputation for the remaining missing values ---')
    df = hot_deck_imputation(df, num_vars + cat_vars, k=k)
    return df


def remove_outliers(df, variables, iforest_kws=None):
    """
    Removes outliers using the Isolation Forest algorithm
    (https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.IsolationForest.html).

    Parameters
    ----------
    df : `pandas.DataFrame`
        DataFrame containing the data.
    variables : list
        Variables with potential outliers.
    iforest_kws : dict, default=None
        IsolationForest algorithm hyperparameters.

    Returns
    ----------
    df_inliers : `pandas.DataFrame`
        DataFrame with inliers (i.e. observations that are not outliers).
    df_outliers : `pandas.DataFrame`
        DataFrame with outliers.
    """
    if iforest_kws is None:
        iforest_kws = dict(max_samples=0.8, max_features=0.8, bootstrap=False, random_state=42)
    outlier_if = IsolationForest(**iforest_kws)
    outlier_flag = outlier_if.fit_predict(df[variables])
    return df[outlier_flag > 0], df[outlier_flag < 0]
