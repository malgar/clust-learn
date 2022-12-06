"""Table statistics utils for clustering"""
# Author: Miguel Alvarez-Garcia


def compare_cluster_means_to_global_means(df, dimensions, data_standardized=False, output_path=None):
    df_agg = df.groupby('cluster')[dimensions].mean()
    df_agg_diff = df_agg.copy()
    if data_standardized:
        df_agg_diff = df_agg
    else:
        mean_array = df[dimensions].mean().values
        for idx, row in df_agg.iterrows():
            df_agg_diff.loc[idx, dimensions] = (row[dimensions] - mean_array) / mean_array

    df_agg_diff = df_agg_diff.reset_index()
    if output_path is not None:
        df_agg_diff.to_csv(output_path, index=False)

    return df_agg_diff
