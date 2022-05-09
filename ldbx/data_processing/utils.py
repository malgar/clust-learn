# Utils for daya processing

import networkx as nx
import numpy as np
import pandas as pd

from scipy.stats import rv_discrete
from sklearn.feature_selection import mutual_info_regression, mutual_info_classif
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mutual_info_score
from ..utils import cross_corr_ratio


def compute_missing(df):
    missing_df = (df.isnull().sum()*100/df.shape[0]).to_frame('pct_missing').reset_index()\
        .rename(columns={'index': 'var_name'})
    missing_df = missing_df.sort_values('pct_missing', ascending=False)
    return missing_df


def num_imputation_pairs(df, corr_thres=0.7, method='pearson'):
    corr_df = df.corr(method=method)
    # We only keep the pairs with a corr coefficient above the threshold and melt the DataFrame to have a row per pair
    num_pairs = corr_df.replace({1: 0})[np.abs(corr_df.replace({1: 0})) > corr_thres].reset_index()\
        .melt(id_vars='index', value_vars=corr_df.columns[1:]).dropna().sort_values('value', ascending=False) \
        .rename(columns={'index': 'var1', 'variable': 'var2'}).reset_index(drop=True)
    return num_pairs


def mixed_imputation_pairs(df, num_vars, cat_vars, cross_corr_thres=0.5):
    cross_corr_df = cross_corr_ratio(df[cat_vars], df[num_vars])
    # We keep only the pairs with a cross correlation coefficient above the threshold and melt the notebook to have a
    # row per pair
    pairs = cross_corr_df[cross_corr_df > cross_corr_thres].reset_index()\
        .melt(id_vars='index', value_vars=cross_corr_df.columns[1:]) .dropna()\
        .sort_values('value', ascending=False).rename(columns={'index': 'var1', 'variable': 'var2'})\
        .reset_index(drop=True)

    # For every pair <cat_var, num_var>, we're interested in the relationship in both ways
    pairs = pd.concat([pairs, pairs.rename(columns={'var1': 'var2', 'var2': 'var1'})], ignore_index=True)
    pairs = pairs.sort_values(['value', 'var1', 'var2'], ascending=[False, True, True]).reset_index(drop=True)
    return pairs


def cat_imputation_pairs(df, mi_thres=0.6):
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


def imputation_pairs(df, num_vars, cat_vars, num_kws=None, mixed_kws=None, cat_kws=None):
    # Numerical variable pairs (correlation)
    num_kws = num_kws if num_kws else dict()
    num_pairs = num_imputation_pairs(df[num_vars], **num_kws)

    # Mixed variable pairs (cross correlation)
    mixed_kws = mixed_kws if mixed_kws else dict()
    mixed_pairs = mixed_imputation_pairs(df, num_vars, cat_vars, **mixed_kws)

    # Categorical variable pairs (mutual information)
    cat_kws = cat_kws if cat_kws else dict()
    cat_pairs = cat_imputation_pairs(df[cat_vars], **cat_kws)

    final_pairs = pd.concat([num_pairs, mixed_pairs, cat_pairs])
    final_pairs = final_pairs.drop(columns='value')

    # We only want to keep those pairs with missing values
    missing_df = compute_missing(df[num_vars + cat_vars])
    final_pairs = final_pairs.merge(missing_df, left_on='var1', right_on='var_name').rename(
        columns={'pct_missing': 'pct_missing_var1'}).drop(columns='var_name')
    final_pairs = final_pairs.merge(missing_df, left_on='var2', right_on='var_name').rename(
        columns={'pct_missing': 'pct_missing_var2'}).drop(columns='var_name')
    # Each variable with missing value and strongly related to at least another one, is assigned to the one from which
    # a larger proportion of missing values can be imputed.
    final_pairs = final_pairs.loc[final_pairs.groupby('var1')['pct_missing_var2'].idxmin()].sort_values(
        ['pct_missing_var2', 'pct_missing_var1'])
    final_pairs = final_pairs[final_pairs['pct_missing_var1'] > final_pairs['pct_missing_var2']].reset_index(drop=True)
    return final_pairs


def bucket(values, q=10):
    b_values = values
    if len(np.unique(values)) > 10:
        b_values = pd.qcut(values, q=q, duplicates='drop')
    return b_values


def impute_missing_values_with_highly_related_pairs(df, num_vars, cat_vars, imputation_pairs):
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
            # Bucket the numerical variable to treat both as categorical.
            if row['var2'] in num_vars:
                # print(f"Bucket variable {row['var2']}")
                df[f"b_{row['var2']}"] = bucket(df[row['var2']])
                df[f"b_{row['var1']}"] = df[row['var1']]
            elif row['var1'] in num_vars:
                # print(f"Bucket variable {row['var1']}")
                df[f"b_{row['var1']}"] = bucket(df[row['var1']])
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
                dist_freq = df.loc[
                    (df[f"b_{row['var2']}"] == pv) & (~df[row['var1']].isnull()), row['var1']].value_counts(
                    normalize=True).sort_index()
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


def mutual_information_pair_scores(df, num_vars, cat_vars):
    # First, we calculate max mutual information per variable for normalization
    max_mi_scores = dict()
    for nv in num_vars:
        values = df[nv].dropna().values
        mi = mutual_info_regression(values.reshape(-1, 1), values, discrete_features=False)[0]
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
                                            discrete_features=False)[0],
                     mutual_info_regression(df_f[num_vars[j]].values.reshape(-1, 1), df_f[num_vars[i]],
                                            discrete_features=False)[0])
            norm_factor = max(max_mi_scores[num_vars[i]], max_mi_scores[num_vars[j]])
            mi = mi / norm_factor
            mi_scores.append((num_vars[i], num_vars[j], mi))

    # Mixed pairs
    for i in range(len(num_vars)):
        for j in range(len(cat_vars)):
            df_f = df[[num_vars[i], cat_vars[j]]].dropna()
            mi = max(mutual_info_classif(df_f[num_vars[i]].values.reshape(-1, 1), df_f[cat_vars[j]],
                                         discrete_features=False)[0],
                     mutual_info_regression(df_f[cat_vars[j]].values.reshape(-1, 1), df_f[num_vars[i]],
                                            discrete_features=True)[0])
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


# TODO: It shouldn't be compulsory to have both types of variables. (This applies to all the modules)
def variable_graph_partitioning(pair_scores, thres=0.5):
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
