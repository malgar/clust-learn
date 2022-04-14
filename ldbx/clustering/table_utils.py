# Table statistics utils for clustering

def compare_cluster_means_to_global_means(df, dimensions, output_path=None):
    df_agg = df.groupby('cluster_cat')[[dimensions]].mean()
    df_agg_diff = df_agg.copy()
    mean_array = df[dimensions].mean().values
    for idx, row in df_agg.iterrows():
        df_agg_diff.loc[idx, dimensions] = (row[dimensions] - mean_array) / mean_array
    df_agg_diff = df_agg_diff.reset_index()

    if output_path is not None:
        df_agg_diff.to_csv(output_path, index=False)

    return df_agg_diff
