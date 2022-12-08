"""Clustering with KMeans++ as default algorithm"""
# Author: Miguel Alvarez-Garcia

import pandas as pd
import statsmodels.api as sm

from scipy.stats import chi2_contingency
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score
from sklearn.preprocessing import MinMaxScaler
from statsmodels.formula.api import ols
from .table_utils import *
from .utils import *
from .viz_utils import *

__metrics__ = ['inertia', 'davies_bouldin_score', 'silhouette_score', 'calinski_harabasz_score']
METRIC_NAMES = dict(zip(__metrics__,
                        ['Weighted Sum of Squared Distances', 'Davies Bouldin Score', 'Silhouette Score',
                         'Calinski and Harabasz score']))
KMEANS = ['kmeans', 'kmeans++']
HIERARCHICAL_WARD = ['ward', 'hierarchical', 'agglomerative']


class Clustering:
    """
    Clustering class

    Parameters
    ----------
    df : `pandas:DatasFrame`
        DataFrame with main data
    algorithms : str or list, default='kmeans'
        Algorithm/s to be used for clustering.
        By default, [K-Means++](https://scikit-learn.org/stable/modules/clustering.html#k-means)
    normalize : bool, default=True
        Whether to apply data normalization for fair comparisons between variables. By default it is applied. In case
        dimensionality reduction is applied beforehand, normalization should not be applied.
    """
    def __init__(self,
                 df,
                 algorithms='kmeans',
                 normalize=True):

        # Normalize variables for fair comparisons
        self.normalize = normalize
        self.df = df.copy()
        if normalize:
            mms = MinMaxScaler()
            self.df = pd.DataFrame(mms.fit_transform(df), columns=df.columns)

        self.dimensions_ = list(df.columns)

        if not isinstance(algorithms, list):
            algorithms = [algorithms]
        self.algorithms = list(map(str.lower, algorithms))

        self.instances_ = dict()
        for algorithm in self.algorithms:
            if algorithm in KMEANS:
                self.instances_[algorithm] = KMeans(random_state=42)
            elif algorithm in HIERARCHICAL_WARD:
                self.instances_[algorithm] = AgglomerativeClustering()
            else:
                raise RuntimeWarning(f'''Algorithm {algorithm} is not supported.
                                     Supported algorithms are: KMeans and Hierarchical clustering''')

        self.scores_ = dict()
        self._initialize_scores()

        self.metric_ = 'inertia'
        self.optimal_config_ = None

    def _initialize_scores(self):
        for algorithm in self.algorithms:
            if algorithm in KMEANS + HIERARCHICAL_WARD:
                self.scores_[algorithm] = []

    def _compute_clusters(self, algorithm, n_clusters):
        self.instances_[algorithm].set_params(n_clusters=n_clusters)
        self.instances_[algorithm].fit(self.df[self.dimensions_])
        self.labels_ = self.instances_[algorithm].labels_

    def _compute_optimal_clustering_config(self, metric, cluster_range, weights):
        optimal_list = []
        for algorithm in self.algorithms:
            for nc in range(*cluster_range):
                self.instances_[algorithm].set_params(n_clusters=nc)
                self.instances_[algorithm].fit(self.df[self.dimensions_])

                if metric == 'inertia':
                    self.scores_[algorithm].append(
                        weighted_sum_of_squared_distances(self.df[self.dimensions_], self.instances_[algorithm].labels_,
                                                          weights))
                elif metric == 'davies_bouldin_score':
                    self.scores_[algorithm].append(
                        1 if nc == 1 else davies_bouldin_score(self.df[self.dimensions_],
                                                               self.instances_[algorithm].labels_))
                elif metric == 'silhouette_score':
                    self.scores_[algorithm].append(
                        0 if nc == 1 else silhouette_score(self.df[self.dimensions_],
                                                           self.instances_[algorithm].labels_))

                elif metric == 'calinski_harabasz_score':
                    self.scores_[algorithm].append(
                        1 if nc == 1 else calinski_harabasz_score(self.df[self.dimensions_],
                                                                  self.instances_[algorithm].labels_))

            if len(range(*cluster_range)) > 1:
                kl = KneeLocator(x=range(*cluster_range), y=self.scores_[algorithm], curve='convex',
                                 direction='decreasing')
                optimal_list.append((algorithm, kl.knee, self.scores_[algorithm][kl.knee-cluster_range[0]]))
            else:
                optimal_list.append((algorithm, cluster_range[0], self.scores_[algorithm][0]))

        return min(optimal_list, key=lambda t: t[2])

    def compute_clusters(self, n_clusters=None, metric='inertia', max_clusters=10, prefix=None, weights=None):
        """
        Calculates clusters.
        If more than one algorithm is passed in the class constructor, first, the optimal number of clusters
        is computed for each algorithm based on the metric passed to the method. Secondly, the algorithm that
        provides the best performance for the corresponding optimal number of clusters is selected.
        Therefore, the result shows the clusters calculated with the best performing algorithm based on the
        criteria explained above.

        Parameters
        ----------
        n_clusters : int, default=None
            Number of clusters to be computed. If n_clusers=None, the optimal number of clusters is computed
            using the elbow/knee method.
            For optimal number of cluster calculation, the Python package [kneed](https://pypi.org/project/kneed/)
            is used, where the method presented in [this paper](https://raghavan.usc.edu//papers/kneedle-simplex11.pdf)
            is implemented.
        metric : str, default='inertia'
            Metric to use in order to compare different algorithms and, if applicable,to calculate the optimal number
            of clusters.
            By default, inertia is used. Supported metrics: ['inertia', 'davies_bouldin_score', 'silhouette_score',
            'calinski_harabasz_score']
        max_clusters : int, default=10
            In case of optimal search, this parameter limits the maximum number of clusters allowed.
        prefix : str, default=None
            Used for cluster naming. Naming format: `f'{prefix}_{x}'`
        weights : `numpy.array`, default=None
            In case observations have different weights.
            *Note this is not implemented yet.*

        Returns
        ----------
        labels_ : `numpy.array`
            Cluster label of each observation.
        """
        if metric not in __metrics__:
            raise RuntimeError(f'''Metric {metric} not supported.
                               Supported metrics: {__metrics__}''')

        self._initialize_scores()
        self.metric_ = metric

        # Compute optimal number of clusters
        cluster_range = []
        if n_clusters is not None:
            cluster_range = [n_clusters, n_clusters + 1]
        else:
            cluster_range = [1, max_clusters + 1]

        self.optimal_config_ = self._compute_optimal_clustering_config(metric, cluster_range, weights)

        if self.optimal_config_ is None:
            raise RuntimeError('Optimal cluster configuration not available')

        self._compute_clusters(self.optimal_config_[0], self.optimal_config_[1])

        # Might be interesting to only keep one
        self.df['cluster'] = self.labels_
        self.df['cluster_cat'] = self.labels_
        if prefix is not None:
            name_len = len(str(max(self.labels_)))
            self.df['cluster_cat'] = list(map(lambda x: f'{prefix}_{str.zfill(str(x), name_len)}', self.labels_))

        return self.labels_

    def describe_clusters(self, df_ext=None, variables=None, cluster_filter=None, statistics=['mean', 'median', 'std'],
                          output_path=None):
        """
        Describes clusters based on internal or external *continuous* variables.
        For categorical variables use `describe_clusters_cat()`.

        Parameters
        ----------
        df_ext : `pandas.DataFrame`, default=None
            Optional. For describing clusters based on external variables.
            This DataFrame must only contain the variables of interest. The order of the observations must be the same
            as that of the base DataFrame.
        variables : str or list, default=None
            List of variables (internal or external) for describing clusters. This parameter is optional and should
            be used when only a subset of the variable is of interest.
        cluster_filter : str or list, default=None
            In case the descriptive statistics of interest only applies to a subset of the calculated clusters.
        statistics : str or list, default=['mean', 'median', 'std']
            Statistics to use for describing clusters.
            *Note any statistics supported by Pandas can be used. This includes the `describe` function*
        output_path : str, default=None
            If an output_path is passed, the resulting DataFame is saved as a CSV file.

        Returns
        ----------
        res : `pandas.DataFrame`
            DataFrame with the cluster description.
        """
        if df_ext is not None:
            df_ext = df_ext.copy()
            df_ext['cluster'] = self.labels_
        else:
            df_ext = self.df[self.dimensions_ + ['cluster']]

        if variables is not None:
            if not isinstance(variables, list):
                variables = [variables]
            df_ext = df_ext[variables + ['cluster']]
        else:
            variables = df_ext.columns.to_list()
            variables.remove('cluster')

        if cluster_filter is None:
            cluster_filter = list(df_ext['cluster'].unique())
        if not isinstance(cluster_filter, list):
            cluster_filter = [cluster_filter]

        res = df_ext[df_ext['cluster'].isin(cluster_filter)].groupby('cluster').agg(
            dict(zip(list(variables), [statistics] * len(variables)))).reset_index()

        if output_path is not None:
            res.to_csv(output_path, index=False)

        return res

    def describe_clusters_cat(self, cat_array, cat_name=None, order=None, normalize=False, output_path=None):
        """
        Describes clusters based on  external *categorical* variables. The result is a contingency table.
        For continuous variables use `describe_clusters()`.

        Parameters
        ----------
        cat_array : `pandas.Series` or `numpy.array`
            Values of categorical variable.
            The order of the observations must be the same as that of the base DataFrame.
        cat_name : str, default=None
            Name of the categorical variable.
        order : list or `numpy.array`, default=None
            In case categories should be displayed in a specific order.
        normalize : boolean, default=False
            If True, results are row-normalized.
        output_path : str, default=None
            If an output_path is passed, the resulting DataFame is saved as a CSV file.

        Returns
        ----------
        freq : `pandas.DataFrame`
            DataFrame with a cluster description based on the passed categorical variable.
        """
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

    def compare_cluster_means_to_global_means(self, df_original=None, output_path=None):
        """
        For every cluster and every internal variable, the relative difference between the intra-cluster mean
        and the global mean.

        Parameters
        ----------
        df_original : `pandas.DataFrame`, default=None
            In case the comparison wants to be made with the original variables and values. Note it is assumed that both
            the original dataframe and the one used for clustering have observations is the same order.
        output_path : str, default=None
            If an output_path is passed, the resulting DataFame is saved as a CSV file.

        Returns
        ----------
        df_agg_diff : `pandas.DataFrame`
            DataFrame with the comparison.
        """
        if df_original is not None:
            var_names = list(df_original.columns)
            df_original = df_original.copy()
            df_original['cluster'] = self.labels_
            return compare_cluster_means_to_global_means(df_original, var_names, output_path=output_path)
        else:
            return compare_cluster_means_to_global_means(self.df, self.dimensions_, data_standardized=not self.normalize,
                                                         output_path=output_path)

    def anova_tests(self, df_test=None, vars_test=None, cluster_filter=None, output_path=None):
        """
        Runs ANOVA tests for a given set of continuous variables (internal or external) to test dependency with
        clusters.

        Parameters
        ----------
        df_test : `pandas.DataFrame`, default=None
            Optional. For running tests with external continuous variables.
            This DataFrame must only contain the variables of interest. The order of the observations must be the same
            as that of the base DataFrame.
        vars_test : str, default=None
            List of variables (internal or external) tu run tests on. This parameter is optional and should
            be used when only a subset of the variable is of interest.
        cluster_filter : str or list, default=None
            In case the tests should only be applied on a subset of  clusers.
        output_path : str, default=None
            If an output_path is passed, the resulting DataFame is saved as a CSV file.

        Returns
        ----------
        res : `pandas.DataFrame`
            DataFrame with the corresponding test statistics.
        """
        if df_test is not None:
            df_test = df_test.copy()
            df_test['cluster'] = self.labels_
        else:
            df_test = self.df[self.dimensions_ + ['cluster']]

        if vars_test is not None:
            if not isinstance(vars_test, list):
                vars_test = [vars_test]
            df_test = df_test[vars_test + ['cluster']]

        if cluster_filter is None:
            cluster_filter = list(df_test['cluster'].unique())
        if not isinstance(cluster_filter, list):
            cluster_filter = [cluster_filter]

        res = []
        col_names = []
        variables = df_test.columns.to_list()
        variables.remove('cluster')
        for var in variables:
            model = ols(f'{var} ~ C(cluster)', data=df_test[df_test['cluster'].isin(cluster_filter)]).fit()
            aov_table = sm.stats.anova_lm(model, typ=1)
            res.append([var] + aov_table.iloc[0].to_list())
            if len(col_names) == 0:
                col_names = ['var_name'] + aov_table.columns.tolist()

        res = pd.DataFrame(res, columns=col_names)
        if output_path is not None:
            res.to_csv(output_path, index=False)

        return res

    def chi2_test(self, cat_array):
        """
        Runs Chi-squared tests for a given categorical variable to test dependency with clusters.

        Parameters
        ----------
        cat_array : `pandas.Series` or `numpy.array`
            Values of categorical variable.
            The order of the observations must be the same as that of the base DataFrame.

        Returns
        ----------
        dict : dictionary
            Dictionary with the corresponding test statistics.
        """
        contingency_t = self.describe_clusters_cat(cat_array)
        test_res = chi2_contingency(contingency_t.values)
        return dict(zip(['chi2', 'p', 'dof'], test_res[:-1]))

    def plot_score_comparison(self, output_path=None, savefig_kws=None):
        """
        Plots the comparison in performance between the different clustering algorithms.

        Parameters
        ----------
        output_path : str, default=None
            Path to save figure as image.
        savefig_kws : dict, default=None
            Save figure options.
        """
        metric_name = METRIC_NAMES[self.metric_]

        cluster_range = [1, self.optimal_config_[1] + 1]
        if len(self.scores_[self.optimal_config_[0]]) > 1:
            cluster_range = [1, len(self.scores_[self.optimal_config_[0]]) + 1]
        else:
            cluster_range = [self.optimal_config_[1], self.optimal_config_[1] + 1]

        plot_score_comparison(self.scores_, cluster_range, metric_name, output_path, savefig_kws)

    def plot_optimal_components_normalized(self, output_path=None, savefig_kws=None):
        """
        Plots the normalized curve used for computing the optimal number of clusters.

        Parameters
        ----------
        output_path : str, default=None
            Path to save figure as image.
        savefig_kws : dict, default=None
            Save figure options.
        """
        if len(self.scores_[self.optimal_config_[0]]) > 1:
            plot_optimal_components_normalized(self.scores_[self.optimal_config_[0]],
                                               len(self.scores_[self.optimal_config_[0]]),
                                               METRIC_NAMES[self.metric_],
                                               output_path, savefig_kws)
        else:
            raise RuntimeError('This plot can only be used when `cluster_range` contains at least 2 values')

    def plot_clustercount(self, output_path=None, savefig_kws=None):
        """
        Plots a bar plot with cluster counts.

        Parameters
        ----------
        output_path : str, default=None
            Path to save figure as image.
        savefig_kws : dict, default=None
            Save figure options.
        """
        plot_clustercount(self.df, output_path, savefig_kws)

    def plot_cluster_means_to_global_means_comparison(self, df_original=None, xlabel=None, ylabel=None,
                                                      levels=[-0.50, -0.32, -0.17, -0.05, 0.05, 0.17, 0.32, 0.50],
                                                      output_path=None, savefig_kws=None):
        """
        Plots the normalized curve used for computing the optimal number of clusters.

        Parameters
        ----------
        df_original : `pandas.DataFrame`, default=None
            In case the comparison wants to be made with the original variables and values. Note it is assumed that both
            the original dataframe and the one used for clustering have observations is the same order.
        xlabel : str, default=None
            x-label name/description.
        ylabel : str, default=None
            y-label name/description.
        levels : list or `numpy.array`
            Values to be used as cuts for color intensity.
            Default values: [-0.50, -0.32, -0.17, -0.05, 0.05, 0.17, 0.32, 0.50]
        output_path : str, default=None
            Path to save figure as image.
        savefig_kws : dict, default=None
            Save figure options.
        """

        if df_original is not None:
            var_names = list(df_original.columns)
            df_original = df_original.copy()
            df_original['cluster'] = self.labels_

            plot_cluster_means_to_global_means_comparison(df_original, var_names, xlabel, ylabel, levels,
                                                          data_standardized=False, output_path=output_path,
                                                          savefig_kws=savefig_kws)
        else:
            plot_cluster_means_to_global_means_comparison(self.df, self.dimensions_, xlabel, ylabel, levels,
                                                          data_standardized=not self.normalize, output_path=output_path,
                                                          savefig_kws=savefig_kws)

    def plot_distribution_comparison_by_cluster(self, df_ext=None, xlabel=None, ylabel=None, output_path=None,
                                                savefig_kws=None):
        """
        Plots the violin plots per cluster and *continuous* variables of interest to understand differences in their
        distributions by cluster.

        Parameters
        ----------
        df_ext : `pandas.DataFrame`, default=None
            DataFrame containing external variables for comparison.
            If None, internal variables will be compared.
        xlabel : str, default=None
            x-label name/description.
        ylabel : str, default=None
            y-label name/description.
        output_path : str, default=None
            Path to save figure as image.
        savefig_kws : dict, default=None
            Save figure options.
        """
        sharey = self.normalize
        if df_ext is None:
            df_ext = self.df[self.dimensions_]
        else:
            sharey = False

        plot_distribution_by_cluster(df_ext, self.labels_, xlabel=xlabel, ylabel=ylabel, sharey=sharey,
                                     output_path=output_path, savefig_kws=savefig_kws)

    def plot_clusters_2D(self, coor1, coor2, style_kwargs=dict(), output_path=None, savefig_kws=None):
        """
        Plots two 2D plots:
         - A scatter plot styled by the categorical variable `hue`.
         - A 2D plot comparing cluster centroids and optionally the density area.

        Parameters
        ----------
        coor1 : int, str, or `pandas.Series`
            If int, it represents the id of the internal variable to be used.
            If str, it must be an internal variable name.
            If `pandas.Series`, it is assumed it is an external variable.
        coor2 : int, str, or `pandas.Series`
            If int, it represents the id of the internal variable to be used.
            If str, it must be an internal variable name.
            If `pandas.Series`, it is assumed it is an external variable.
        style_kwargs : dict, default=empty dict
            Dictionary with optional styling parameters.
            List of parameters:
             - palette : matplotlib palette to be used. default='gnuplot'
             - vline_color : color to be used for vertical line (used for plotting x mean value). default='#11A579'
             - hline_color : color to be used for horizontal line (used for plotting y mean value). default='#332288'
             - kdeplot : boolean to display density area of points (using seabonr.kdeplot). default=True
        output_path : str, default=None
            Path to save figure as image.
        savefig_kws : dict, default=None
            Save figure options.
        """
        if isinstance(coor1, int):
            coor1 = self.dimensions_[coor1]
        if isinstance(coor2, int):
            coor2 = self.dimensions_[coor2]

        df_ext = self.df
        if isinstance(coor1, pd.Series):
            if not isinstance(coor2, pd.Series):
                raise TypeError('For external variables, both coor1 and coor2 need to be of type pandas.Series')
            df_ext = pd.DataFrame(data={coor1.name: coor1,
                                        coor2.name: coor2,
                                        'cluster_cat': self.df['cluster_cat'].values})
            coor1 = coor1.name
            coor2 = coor2.name

        hue = 'cluster_cat'
        plot_clusters_2D(coor1, coor2, hue, df_ext, style_kwargs=style_kwargs, output_path=output_path,
                         savefig_kws=savefig_kws)

    def plot_cat_distribution_by_cluster(self, cat_array, cat_label=None, cluster_label=None, output_path=None,
                                         savefig_kws=None):
        """
        Plots the relative contingency table of the clusters with a categorical variable as a stacked bar plot.

        Parameters
        ----------
        cat_array : `numpy.array` or list
            Array with categorical values.
            *Note its length must be the same as self.df and observations be in the same order*.
        cat_label : str, default=None
            Name/Description of the categorical variable to be displayed.
        cluster_label : str, default=None
            Name/Description of the cluster variable to be displayed.
        output_path : str, default=None
            Path to save figure as image.
        savefig_kws : dict, default=None
            Save figure options.
        """
        ct = self.describe_clusters_cat(cat_array, normalize=True)
        plot_cat_distribution_by_cluster(ct, cat_label, cluster_label, output_path, savefig_kws)
