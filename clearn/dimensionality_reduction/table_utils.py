"""Table utils for dimensionality reduction"""
# Author: Miguel Alvarez-Garcia

import numpy as np
import pandas as pd

from ..utils import cross_corr_ratio


def cross_corr(df1, df2):
    """
    Calculates the correlation coefficient of every column in df1 with every column in df2

    Parameters
    ----------
    df1 : `pandas.DataFrame`
    df2 : `pandas.DataFrame`

    Returns
    ----------
    corr_df: `pandas.DataFrame`
        DataFrame with the correlation coefficient of every pair of columns from df1 and df2.
    """
    corr_coeffs = []
    for col in df1.columns:
        corr_coeffs.append(df2.corrwith(df1[col]).tolist())

    corr_df = pd.DataFrame(corr_coeffs, index=df1.columns, columns=df2.columns).transpose()
    return corr_df


def num_main_contributors(df, df_trans, thres=0.5, n_contributors=None, dim_idx=None, component_description=None,
                          col_description=None, output_path=None):
    """
    Computes the original numerical variables with the strongest relation to the derived variable(s)
    (measured as Pearson correlation coefficient)

    Parameters
    ----------
    df : `pandas.DataFrame`
        DataFrame with original numerical variables.
    df_trans : `pandas.DataFrame`
        DataFrame with derived variables.
    thres : float, default=0.5
        Correlation coefficient threshold to consider one original variable to be a main contributor of a derived
        variable.
    n_contributors : float, default=None
        If n_contributors is passed, the n_contributors original variables with the highest correlation coefficient
        are selected for every derived variable.
        If n_contributors is passed, the correlation coefficient threshold (thres) is ignored.
    dim_idx : int, default=None
        In case only main contributors for derived variable in column position dim_idx are retrieved (starts at 0).
    component_description: str or list
        Description of derived variables. It might be of interest to show a description of the new variables
        on a table for explainability purposes.
    col_description : `pandas.DataFrame`
        DataFrame with two columns: First one with original variable names, and a second one with the description.
        This is also used for explainability purposes.
    output_path : str, default=None
        If an output_path is passed, the resulting DataFame is saved as a CSV file.

    Returns
    ----------
    mc : `pandas.DataFrame`
        DataFrame with the main contributors of every derived variable.
    """
    if dim_idx is not None:
        df_trans = df_trans[df_trans.columns[dim_idx]].to_frame()

    if n_contributors is not None:
        thres = 0

    corrs = cross_corr(df, df_trans)
    mc = pd.DataFrame()
    for idx, row in corrs.iterrows():
        pc_corrs = row[(row < -thres) | (row > thres)].to_frame().reset_index().rename(
            columns={'index': 'var_name', idx: 'corr_coeff'})
        pc_corrs['component'] = idx
        pc_corrs['corr_coeff_abs'] = np.abs(pc_corrs['corr_coeff'])
        pc_corrs = pc_corrs.sort_values('corr_coeff_abs', ascending=False).drop(columns='corr_coeff_abs')

        if n_contributors is not None:
            pc_corrs = pc_corrs.iloc[:n_contributors]

        mc = pd.concat([mc, pc_corrs], ignore_index=True)

    final_cols = ['component', 'var_name', 'corr_coeff']

    if component_description is not None:
        if not isinstance(component_description, list):
            component_description = [component_description]

        component_description = pd.DataFrame(data={'component': mc['component'].unique(),
                                                   'component_description': component_description})
        final_cols.insert(1, 'component_description')
        mc = mc.merge(component_description, on='component')

    if col_description is not None:
        if not isinstance(col_description, pd.DataFrame):
            raise RuntimeError('`col_description` must be an instance of pandas.DataFrame')

        col_description.columns = ['var_name', 'var_description']
        final_cols.insert(final_cols.index('var_name')+1, 'var_description')
        mc = mc.merge(col_description, on='var_name')

    if output_path is not None:
        mc[final_cols].to_csv(output_path, index=False)

    return mc[final_cols]


def cat_main_contributors(df, df_trans, thres=0.14, n_contributors=None, dim_idx=None, component_description=None,
                          col_description=None, output_path=None):
    """
    Computes the original categorical variables with the strongest relation to the derived variable(s)
    (measured as correlation ratio)

    Parameters
    ----------
    df : `pandas.DataFrame`
        DataFrame with original categorical variables.
    df_trans : `pandas.DataFrame`
        DataFrame with derived variables.
    thres : float, default=0.14
        Correlation ratio threshold to consider one original variable to be a main contributor of a derived
        variable.
    n_contributors : float, default=None
        If n_contributors is passed, the n_contributors original variables with the highest correlation ratio
        are selected for every derived variable.
        If n_contributors is passed, the correlation ratio threshold (thres) is ignored.
    dim_idx : int, default=None
        In case only main contributors for derived variable in column position dim_idx are retrieved (starts at 0).
    component_description : str or list
        Description of derived variables. It might be of interest to show a description of the new variables
        on a table for explainability purposes.
    col_description : `pandas.DataFrame`
        DataFrame with two columns: First one with original variable names, and a second one with the description.
        This is also used for explainability purposes.
    output_path : str, default=None
        If an output_path is passed, the resulting DataFame is saved as a CSV file.

    Returns
    ----------
    mc : `pandas.DataFrame`
        DataFrame with the main contributors of every derived variable.
    """
    if dim_idx is not None:
        df_trans = df_trans[df_trans.columns[dim_idx]].to_frame()

    if n_contributors is not None:
        thres = 0

    corrs = cross_corr_ratio(df, df_trans)
    mc = pd.DataFrame()
    for idx, row in corrs.iterrows():
        pc_corrs = row[row > thres].to_frame().reset_index().rename(columns={'index': 'var_name', idx: 'corr_ratio'})
        pc_corrs['component'] = idx
        pc_corrs = pc_corrs.sort_values('corr_ratio', ascending=False).reset_index(drop=True)

        if n_contributors is not None:
            pc_corrs = pc_corrs.iloc[:n_contributors]

        mc = pd.concat([mc, pc_corrs], ignore_index=True)

    final_cols = ['component', 'var_name', 'corr_ratio']

    if component_description is not None:
        if not isinstance(component_description, list):
            component_description = [component_description]

        component_description = pd.DataFrame(data={'component': mc['component'].unique(),
                                                   'component_description': component_description})
        final_cols.insert(1, 'component_description')
        mc = mc.merge(component_description, on='component')

    if col_description is not None:
        if not isinstance(col_description, pd.DataFrame):
            raise RuntimeError('`col_description` must be an instance of pandas.DataFrame')

        col_description.columns = ['var_name', 'var_description']
        final_cols.insert(final_cols.index('var_name')+1, 'var_description')
        mc = mc.merge(col_description, on='var_name')

    if output_path is not None:
        mc[final_cols].to_csv(output_path, index=False)

    return mc[final_cols]


def cat_main_contributors_stats(df, df_trans, thres=0.14, n_contributors=None, dim_idx=None, output_path=None):
    """
    Computes for every categorical variable's value, the mean and std of the derived variables that are strongly
    related to the categorical variable (based on the correlation ratio)

    Parameters
    ----------
    df : `pandas.DataFrame`
        DataFrame with original categorical variables.
    df_trans : `pandas.DataFrame`
        DataFrame with derived variables.
    thres : float, default=0.14
        Correlation ratio threshold to consider one original variable to be a main contributor of a derived
        variable.
    n_contributors : float, default=None
        If n_contributors is passed, the n_contributors original variables with the highest correlation ratio
        are selected for every derived variable.
        If n_contributors is passed, the correlation ratio threshold (thres) is ignored.
    dim_idx : int, default=None
        In case only main contributors for derived variable in column position dim_idx are retrieved (starts at 0).
    output_path : str, default=None
        If an output_path is passed, the resulting DataFame is saved as a CSV file.

    Returns
    ----------
    stats : `pandas.DataFrame`
        DataFrame with the statistics.
    """
    if dim_idx is not None:
        df_trans = df_trans[df_trans.columns[dim_idx]].to_frame()

    m1 = pd.melt(df.reset_index().rename(columns={'index': 'id'}), id_vars=['id'], value_vars=df.columns)
    m2 = pd.melt(df_trans.reset_index().rename(columns={'index': 'id'}), id_vars=['id'], value_vars=df_trans.columns)
    stats = m1.merge(m2, on='id')
    stats = stats.groupby(['variable_y', 'variable_x', 'value_x']).agg({'value_y': ['mean', 'std']}) \
        .reset_index().rename(columns={'variable_y': 'component', 'variable_x': 'var_name', 'value_x': 'var_value'})
    stats.columns = [c[1] if c[1] != '' else c[0] for c in stats.columns]
    stats = stats.rename(columns={'mean': 'component_mean', 'std': 'component_std'})

    mc = cat_main_contributors(df, df_trans, thres=thres, n_contributors=n_contributors, dim_idx=dim_idx)
    stats = mc[['component', 'var_name']].merge(stats, on=['component', 'var_name'])

    if output_path is not None:
        stats.to_csv(output_path, index=False)

    return stats
