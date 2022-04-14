"""Clustering with KMeans++ as default algorithm"""
# Author: Miguel Alvarez

import pandas as pd
import statsmodels.api as sm

from kneed import KneeLocator
from scipy.stats import chi2_contingency
from sklearn import base
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.metrics import davies_bouldin_score, silhouette_score
from statsmodels.formula.api import ols
from utils import *
from viz_utils import *

__metrics__ = ['inertia', 'davies_bouldin_score', 'silhouette_score']
METRIC_NAMES = dict(zip(__metrics__,
                        ['Weighted Sum of Squared Distances', 'Davies Bouldin Score', 'Silhouette Score']))
KMEANS = ['kmeans', 'kmeans++']
HIERARCHICAL_WARD = ['ward', 'hierarchical', 'agglomerative']


class Clustering(base.BaseEstimator, base.TransformerMixin):
    """
    Clustering class

    Parameters
    ----------
    algorithms : str or list, default='kmeans'
        Algorithm/s to be used for clustering.
        By default, [K-Means++](https://scikit-learn.org/stable/modules/clustering.html#k-means)
    """
    def __init__(self,
                 df,
                 algorithms='kmeans'):

        # TODO: Normalize data
        self.df = df
        self.dimensions = list(df.columns)

        if not isinstance(algorithms, list):
            algorithms = [algorithms]
        self.algorithms = list(map(str.lower, algorithms))

        self.instances = dict()
        for algorithm in self.algorithms:
            if algorithm in KMEANS:
                self.instances[algorithm] = KMeans(random_state=42)
            elif algorithm in HIERARCHICAL_WARD:
                self.instances[algorithm] = AgglomerativeClustering()
            else:
                raise RuntimeWarning(f'''Algorithm {algorithm} is not supported.
                                     Supported algorithms are: KMeans and Hierarchical clustering''')

        self.scores = dict()
        for algorithm in self.algorithms:
            if algorithm in KMEANS + HIERARCHICAL_WARD:
                self.scores[algorithm] = []

        self.metric = 'inertia'
        self.optimal_config = None

    def _compute_clusters(self, algorithm, n_clusters):
        self.instances[algorithm].set_params(n_clusters=n_clusters)
        self.instances[algorithm].fit(self.df[self.dimensions])
        self.labels_ = self.instances[algorithm].labels_

    def _compute_optimal_clustering_config(self, metric, cluster_range, weights):
        optimal_list = []
        for algorithm in self.algorithms:
            for nc in range(*cluster_range):
                self.instances[algorithm].set_params(n_clusters=nc)
                self.instances[algorithm].fit(self.df[self.dimensions])

                if metric == 'inertia':
                    self.scores[algorithm].append(
                        weighted_sum_of_squared_distances(self.df[self.dimensions], self.instances[algorithm].labels_,
                                                          weights))
                elif metric == 'davies_bouldin_score':
                    self.scores[algorithm].append(
                        1 if nc == 1 else davies_bouldin_score(self.df[self.dimensions],
                                                               self.instances[algorithm].labels_))
                elif metric == 'silhouette_score':
                    self.scores[algorithm].append(
                        0 if nc == 1 else silhouette_score(self.df[self.dimensions], self.instances[algorithm].labels_))

            if len(range(*cluster_range)) > 1:
                kl = KneeLocator(x=range(*cluster_range), y=self.scores[algorithm], curve='convex',
                                 direction='decreasing')
                optimal_list.append((algorithm, kl.knee, self.scores[algorithm][kl.knee-cluster_range[0]]))
            else:
                optimal_list.append((algorithm, cluster_range[0], self.scores[algorithm][0]))

        return min(optimal_list, key=lambda t: t[2])

    def compute_clusters(self, n_clusters=None, metric='inertia', max_clusters=10, prefix=None, weights=None):
        if metric not in __metrics__:
            raise RuntimeError(f'''Metric {metric} not supported.
                               Supported metrics: {__metrics__}''')

        self.metric = metric

        # Compute optimal number of clusters
        cluster_range = []
        if n_clusters is not None:
            cluster_range = [n_clusters, n_clusters + 1]
        else:
            cluster_range = [1, max_clusters + 1]

        self.optimal_config = self._compute_optimal_clustering_config(metric, cluster_range, weights)

        if self.optimal_config is None:
            raise RuntimeError('Optimal cluster configuration not available')

        self._compute_clusters(self.optimal_config[0], self.optimal_config[1])

        self.df['cluster'] = self.labels_
        self.df['cluster_cat'] = self.labels_
        if prefix is not None:
            self.df['cluster_cat'] = list(map(lambda x: f'{prefix}_{x}', self.labels_))

        return self.labels_

    def describe_clusters(self, df_ext=None, variables=None, cluster_labels=None, statistics=['mean', 'median', 'std'],
                          output_path=None):
        if df_ext is not None:
            df_ext['cluster'] = self.labels_
        else:
            df_ext = self.df[self.dimensions + ['cluster']]

        if variables is not None:
            if not isinstance(variables, list):
                variables = [variables]
            df_ext = df_ext[variables + ['cluster']]
        else:
            variables = df_ext.columns.to_list()
            variables.remove('cluster')

        if cluster_labels is None:
            cluster_labels = df_ext['cluster'].unique()
        if not isinstance(cluster_labels, list):
            cluster_labels = [cluster_labels]

        res = df_ext[df_ext['cluster'].isin(cluster_labels)].groupby('cluster').agg(
            dict(zip(list(variables), [statistics] * len(variables)))).reset_index()

        if output_path is not None:
            res.to_csv(output_path, index=False)

        return res

    def describe_clusters_cat(self, cat_array, cat_name=None, order=None, normalize=False, output_path=None):
        freq = pd.crosstab(index=self.df['cluster_cat'], columns=cat_array, rownames=['Clusters'], colnames=[cat_name])
        if order is not None:
            freq = freq[order]

        if normalize:
            freq['total'] = freq.sum(1)
            for col in freq.columns[:-1]:
                freq[col] = freq[col] / freq['total']
            freq = freq.drop(columns='total')

        if output_path is not None:
            freq.to_csv(output_path, index=False)

        return freq

    def compare_cluster_means_to_global_means(self, output_path=None):
        df_agg = self.df.groupby('cluster_cat')[[self.dimensions]].mean()
        df_agg_diff = df_agg.copy()
        mean_array = self.df[self.dimensions].mean().values
        for idx, row in df_agg.iterrows():
            df_agg_diff.loc[idx, self.dimensions] = (row[self.dimensions] - mean_array) / mean_array
        df_agg_diff = df_agg_diff.reset_index()

        if output_path is not None:
            df_agg_diff.to_csv(output_path, index=False)

        return df_agg_diff

    # TODO This allows for internal and external use
    # TODO: vars_test can be str or list, same as cluster_labels
    def anova_tests(self, df_test=None, vars_test=None, cluster_labels=None, output_path=None):
        if df_test is not None:
            df_test['cluster'] = self.labels_
        else:
            df_test = self.df[self.dimensions + ['cluster']]

        if vars_test is not None:
            if not isinstance(vars_test, list):
                vars_test = [vars_test]
            df_test = df_test[vars_test + ['cluster']]

        if cluster_labels is None:
            cluster_labels = df_test['cluster'].unique()
        if not isinstance(cluster_labels, list):
            cluster_labels = [cluster_labels]

        res = []
        col_names = []
        variables = df_test.columns.to_list()
        variables.remove('cluster')
        for var in variables:
            model = ols(f'{var} ~ C(cluster)', data=df_test[df_test['cluster'].isin(cluster_labels)]).fit()
            aov_table = sm.stats.anova_lm(model, typ=1)
            res.append([var] + aov_table.iloc[0].to_list())
            if len(col_names) == 0:
                col_names = ['var_name'] + aov_table.columns.tolist()

        res = pd.DataFrame(res, columns=col_names)
        if output_path is not None:
            res.to_csv(output_path, index=False)

        return res

    def chi2_test(self, cat_array):
        contingency_t = self.describe_clusters_cat(cat_array)
        test_res = chi2_contingency(contingency_t.values)
        return test_res[:-1]

    def plot_score_comparison(self, output_path=None, savefig_kws=None):
        metric_name = METRIC_NAMES[self.metric]

        cluster_range = [1, self.optimal_config[1] + 1]
        if len(self.scores[self.optimal_config[0]]) > 1:
            cluster_range = [1, len(self.scores[self.optimal_config[0]]) + 1]
        else:
            cluster_range = [self.optimal_config[1], self.optimal_config[1] + 1]

        plot_score_comparison(self.scores, cluster_range, metric_name, output_path, savefig_kws)

    def plot_optimal_components_normalized(self, output_path=None, savefig_kws=None):
        if len(self.scores[self.optimal_config[0]]) > 1:
            plot_optimal_components_normalized(self.scores[self.optimal_config[0]],
                                               len(self.scores[self.optimal_config[0]]),
                                               METRIC_NAMES[self.metric],
                                               output_path, savefig_kws)
        else:
            raise RuntimeError('This plot can only be used when `cluster_range` contains at least 2 values')

    def plot_clustercount(self, output_path=None, savefig_kws=None):
        plot_clustercount(self.df, output_path, savefig_kws)

    def plot_cluster_means_to_global_means_comparison(self, xlabel=None, ylabel=None, output_path=None,
                                                      savefig_kws=None):
        plot_cluster_means_to_global_means_comparison(self.df, self.dimensions, xlabel, ylabel, output_path,
                                                      savefig_kws)

    # TODO: df_ext in case we'd like to compare against external variables
    def plot_distribution_comparison_by_cluster(self, df_ext=None, xlabel=None, ylabel=None, output_path=None,
                                                savefig_kws=None):
        if df_ext is None:
            df_ext = self.df[self.dimensions]
        plot_distribution_comparison_by_cluster(df_ext, self.labels_, xlabel, ylabel, output_path, savefig_kws)

    # TODO: coor1 and coor2 can be integer or str. Be careful when documenting
    def plot_clusters_2D(self, coor1, coor2, style_kwargs=dict(), output_path=None, savefig_kws=None):
        if isinstance(coor1, int):
            coor1 = self.dimensions[coor1]
        if isinstance(coor2, int):
            coor2 = self.dimensions[coor2]
        hue = 'cluster_cat'
        plot_clusters_2D(coor1, coor2, hue, self.df, style_kwargs=dict(), output_path=None, savefig_kws=None)

    def plot_cat_distribution_by_cluster(self, cat_array, cat_label=None, cluster_label=None, output_path=None,
                                         savefig_kws=None):
        ct = self.describe_clusters_cat(cat_array, normalize=True)
        plot_cat_distribution_by_cluster(ct, cat_label, cluster_label, output_path, savefig_kws)
