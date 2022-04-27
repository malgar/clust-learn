# Utils for daya processing

import numpy as np
import pandas as pd

from sklearn.metrics import mutual_info_score
from ..utils import cross_corr_ratio


def cont_imputation_pairs(df, corr_thres=0.7, method='pearson'):
    corr_df = df.corr(method=method)
    # We only keep the pairs with a corr coefficient above the threshold and melt the DataFrame to have a row per pair
    cont_pairs = corr_df.replace({1: 0})[np.abs(corr_df.replace({1: 0})) > corr_thres].reset_index()\
        .melt(id_vars='index', value_vars=corr_df.columns[1:]).dropna().sort_values('value', ascending=False) \
        .rename(columns={'index': 'var1', 'variable': 'var2'}).reset_index(drop=True)
    return cont_pairs


def mixed_imputation_pairs(df, cont_vars, cat_vars, cross_corr_thres=0.5):
    cross_corr_df = cross_corr_ratio(df[cat_vars], df[cont_vars])
    # We keep only the pairs with a cross correlation coefficient above the threshold and melt the notebook to have a
    # row per pair
    pairs = cross_corr_df[cross_corr_df > cross_corr_thres].reset_index()\
        .melt(id_vars='index', value_vars=cross_corr_df.columns[1:]) .dropna()\
        .sort_values('value', ascending=False).rename(columns={'index': 'var1', 'variable': 'var2'})\
        .reset_index(drop=True)

    # For every pair <disc_var, cont_var>, we're interested in the relationship in both ways
    pairs = pd.concat([pairs, pairs.rename(columns={'var1': 'var2', 'var2': 'var1'})], ignore_index=True)
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

    cat_pairs = pd.DataFrame(data=data, columns=['var1', 'var2', 'value'])
    return cat_pairs
