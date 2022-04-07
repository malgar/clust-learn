"""Clustering with KMeans++ as default algorithm"""
# Author: Miguel Alvarez

from kneed import KneeLocator
from sklearn import base
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.metrics import davies_bouldin_score, silhouette_score
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

    def describe_clusters(self, statistic='mean'):
        return self.df.groupby('cluster_cat')[self.dimensions].agg(statistic).reset_index()

    def compare_cluster_means_to_global_means(self):
        df_agg = self.df.groupby('cluster_cat')[[self.dimensions]].mean()
        df_agg_diff = df_agg.copy()
        mean_array = self.df[self.dimensions].mean().values
        for idx, row in df_agg.iterrows():
            df_agg_diff.loc[idx, self.dimensions] = (row[self.dimensions] - mean_array) / mean_array
        df_agg_diff = df_agg_diff.reset_index()
        return df_agg_diff

    def anova_tests(self):
        # TODO
        pass

    def plot_score_comparison(self, output_path=None, savefig_kws=None):
        metric_name = METRIC_NAMES[self.metric]

        cluster_range = [1, self.optimal_config[1] + 1]
        if len(self.scores[self.optimal_config[0]]) > 1:
            cluster_range = [1, len(self.scores[self.optimal_config[0]]) + 1]
        else:
            cluster_range = [self.optimal_config[1], self.optimal_config[1] + 1]

        plot_score_comparison(self.scores, cluster_range, metric_name, output_path, savefig_kws)

    def plot_optimal_components_normalized(self):
        if len(self.scores[self.optimal_config[0]]) > 1:
            plot_optimal_components_normalized(self.scores[self.optimal_config[0]],
                                               len(self.scores[self.optimal_config[0]]),
                                               METRIC_NAMES[self.metric])
        else:
            raise RuntimeError('This plot can only be used when `cluster_range` contains at least 2 values')

    def plot_cluster_means_to_global_means_comparison(self, xlabel=None, ylabel=None, output_path=None,
                                                      savefig_kws=None):
        plot_cluster_means_to_global_means_comparison(self.df, self.dimensions, xlabel, ylabel, output_path,
                                                      savefig_kws)
