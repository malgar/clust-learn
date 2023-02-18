<!-- title -->
<div align="center">
  <h3>Clust-learn</h3>
</div>

<!-- Short description -->
<p align="center">
   A Python package for extracting information from large and high-dimensional mixed-type data through explainable cluster analysis.
</p>

<br/>

![clust-learn visualizations](https://github.com/malgar/clust-learn/blob/v0.0.8/images/visualizations.png?raw=true)

<hr>

## Table of contents
1. [Introduction](#user-content-introduction)
2. [Overall architecture](#user-content-architecture)
3. [Implementation](#user-content-implementation)
4. [Installation](#user-content-install)
5. [Version and license information](#user-content-license)
6. [Bug reports and future work](#user-content-future)
7. [User guide & API](#user-content-api)
	1. [Data processing](#user-content-module-preprocessing)
		1. [Data imputation](#user-content-module-preprocessing-imputation)
		2. [Outliers](#user-content-module-preprocessing-outliers)
	2. [Dimensionality reduction](#user-content-module-dimensionality)
		- [DimensionalityReduction class](#DimensionalityReduction_class)
		- [Class methods](#DimensionalityReduction_class_methods)
	3. [Clustering](#user-content-module-clustering)
		- [Clustering class](#Clustering_class)
		- [Class methods](#Clustering_class_methods)
	4. [Classifier](#user-content-module-classifier)
		- [Classifier class](#Classifier_class)
		- [Class methods](#Classifier_class_methods)
8. [Citing](#user-content-citing)

<h2 id="user-content-introduction">
1. Introduction
</h2>

`clust-learn` enables users to run end-to-end explainable cluster analysis to extract information from large and high-dimensional
mixed-type data, and it does so by providing a framework that guides the user through data preprocessing, dimensionality reduction, 
clustering, and classification of the obtained clusters. It is designed to require very few lines of code, and with a strong
focus on explainability.

<h2 id="user-content-architecture">
2. Overall architecture
</h2>

`clust-learn` is organized into four modules, one for each component of the methodological framework presented [here](#user-content-citing): 
* [data_preprocessing](https://github.com/malgar/clust-learn/tree/master/clearn/data_preprocessing)
* [dimensionality_reduction](https://github.com/malgar/clust-learn/tree/master/clearn/dimensionality_reduction)
* [clustering](https://github.com/malgar/clust-learn/tree/master/clearn/clustering)
* [classifier](https://github.com/malgar/clust-learn/tree/master/clearn/classifier)

**Figue 1** shows the package layout with the functionalities covered by each module along with the techniques used, the
explainability strategies available, and the main functions and class methods encapsulating these techniques and
explainability strategies.

<br/>

![clust-learn package structure](https://github.com/malgar/clust-learn/blob/v0.0.8/images/package_structure.png?raw=true)

<br/>

<h2 id="user-content-implementation">
3. Implementation
</h2>

The package is implemented with Python 3.9 using open source libraries. It relies heavily on [pandas](https://pandas.pydata.org/) and
[scikit-learn](https://scikit-learn.org/stable/). Read the complete list of requirements [here](https://github.com/malgar/clust-learn/blob/master/requirements.txt).

It can be installed manually or from pip/PyPI (see Section [4. Installation](#user-content-install)).

<h2 id="user-content-install">
4. Installation
</h2>

The package is on [PyPI](https://pypi.org/project/clust-learn/). Simply run:

```
pip install clust-learn
```

<h2 id="user-content-license">
5. Version and license information
</h2>

* Version: 0.1.2
* Author: Miguel Alvarez-Garcia (malvarez.statistics@gmail.com)
* License: GPLv3 

<h2 id="user-content-future">
6. Bug reports and future work
</h2>

Please report bugs and feature requests through creating a new issue [here](https://github.com/malgar/clust-learn/issues).

<h2 id="user-content-api">
7. User guide & API
</h2>

`clust-learn` is organized into four modules:

1. Data preprocessing
2. Dimensionality reduction
3. Clustering
4. Classifier

**Figue 1** shows the package layout with the functionalities covered by each module along with the techniques used, the explainability strategies available, and the main functions and class methods encapsulating these techniques and explainability strategies.

The four modules are designed to be used sequentially to ensure robust and explainable results. However, each of them is independent and can be used separately to suit different use cases.


<h3 id="user-content-module-preprocessing">
7.i. Data preprocessing
</h3>

Data preprocessing consists of a set of manipulation and transformation tasks performed on the raw data before it is used for its analysis. Although data quality is essential for obtaining robust and reliable results, real-world data is often incomplete, noisy, or inconsistent. Therefore, data preprocessing is a crucial step in any analytical study.

<h4 id="user-content-module-preprocessing-imputation">
7.i.a. Data imputation
</h4>

<h4 id="compute_missing">
compute_missing()
</h4>

```
compute_missing(df, normalize=True)
```

Calculates the pct/count of missing values per column.

**Parameters**

- `df` : `pandas.DataFrame`
- `normalize` : `boolean`, default=`True`

**Returns**

- `missing_df` : `pandas.DataFrame`
	- DataFrame with the pct/counts of missing values per column.
	
<h4 id="missing_values_heatmap">
missing_values_heatmap()
</h4>

```
missing_values_heatmap(df, output_path=None, savefig_kws=None)
```

Plots a heatmap to visualize missing values (light color).

**Parameters**

- `df` : `pandas.DataFrame`
   - DataFrame containing the data.
- `output_path` : `str`, default=`None`
   - Path to save figure as image.
- `savefig_kws` : `dict`, default=`None`
   - Save figure options.

<h4 id="impute_missing_values">
impute_missing_values()
</h4>

```
impute_missing_values(df, num_vars, cat_vars, num_pair_kws=None, mixed_pair_kws=None, cat_pair_kws=None, graph_thres=0.05, k=8, max_missing_thres=0.33)
```

This function imputes missing values following this steps:
1. One-to-one model based imputation for strongly related variables.
2. Cluster based hot deck imputation where clusters are obtained as the connected components of an undirected graph *G=(V,E)*, where *V* is the set of variables and *E* the pairs of variables with mutual information above a predefined threshold.
3. Records with a proportion of missing values above a predefined threshold are discarded to ensure the quality of the hot deck imputation.
4. Hot deck imputation for the remaining missing values considering all variables together.

**Parameters**

- `df` : `pandas.DataFrame`
	- Data frame containing the data with potential missing values.
- `num_vars` : `str`, `list`, `pandas.Series`, or `numpy.array`
	- Numerical variable name(s).
- `cat_vars` : `str`, `list`, `pandas.Series`, or `numpy.array`
	- Categorical variable name(s).
- `{num,mixed,cat}_pair_kws` : `dict`, default=`None`
	- Additional keyword arguments to pass to compute imputation pairs for one-to-one model based imputation, namely:
		- For numerical pairs, `corr_thres` and `method` for setting the correlation coefficient threshold and method. By default, `corr_thres=0.7` and `method='pearson'`.
		- For mixed-type pairs, `np2_thres` for setting the a threshold on partial *eta* square with 0.14 as default value.
		- For categorical pairs, `mi_thres` for setting a threshold on mutual information score. By default, `mi_thres=0.6`.
- `graph_thres` : `float`, default=0.05
	- Threshold to determine if two variables are similar based on mutual information score, and therefore are an edge of the graph from which variable clusters are derived.
- `k` : `int`, default=8
	- Number of neighbors to consider in hot deck imputation.
- `max_missing_thres`: `float`, default=0.33
	- Maximum proportion of missing values per observation allowed before final general hot deck imputation - see step 3 of the missing value imputation methodology in section 2.1.

**Returns**

- `final_pairs` : `pandas.DataFrame`
	- DataFrame with pairs of highly correlated variables (`var1`: variable with values to impute; `var2`: variable to be used as independent variable for model-based imputation), together proportion of missing values of variables `var1` and `var2`.
   
<h4 id="plot_imputation_distribution_assessment">
plot_imputation_distribution_assessment()
</h4>

```
plot_imputation_distribution_assessment(df_prior, df_posterior, imputed_vars, sample_frac=1.0, prior_kws=None, posterior_kws=None, output_path=None, savefig_kws=None)
```

Plots a distribution comparison of each variable with imputed variables, before and after imputation.

**Parameters**

- `df_prior` : `pandas.DataFrame`
	- DataFrame containing the data before imputation.
- `df_posterior` : `pandas.DataFrame`
	- DataFrame containing the data after imputation.
- `imputed_vars` : `list`
	- List of variables with imputed variables.
- `sample_frac` : float, default=1.0
	- If < 1 a random sample of every pair of variables will be plotted.
- `{prior,posterior}_kws` : `dict`, default=`None`
	- Additional keyword arguments to pass to the [kdeplot](https://seaborn.pydata.org/generated/seaborn.kdeplot.html).
- `output_path` : `str`, default=`None`
	- Path to save figure as image.
- `savefig_kws` : `dict`, default=`None`
	- Save figure options.

<h4 id="user-content-module-preprocessing-outliers">
7.i.b. Outliers
</h4>

<h4 id="remove_outliers">
remove_outliers()
</h4>

```
remove_outliers(df, variables, iforest_kws=None)
```

Removes outliers using the [Isolation Forest algorithm](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.IsolationForest.html).

**Parameters**

- `df` : `pandas.DataFrame`
	- DataFrame containing the data.
- `variables` : `list`
	- Variables with potential outliers.
- `iforest_kws` : `dict`, default=`None`
	- IsolationForest algorithm hyperparameters.

**Returns**

- df_inliers : `pandas.DataFrame`
	- DataFrame with inliers (i.e. observations that are not outliers).
- df_outliers : `pandas.DataFrame`
	- DataFrame with outliers.


<h3 id="user-content-module-dimensionality">
7.ii. Dimensionality reduction
</h3>

All the functionality of this module is encapsulated in the `DimensionalityReduction` class so that the original data, the instances of the models used, and any other relevant information is self-maintained and always accessible.

<h4 id="DimensionalityReduction_class">
DimensionalityReduction class
</h4>

```
dr = DimensionalityReduction(df, num_vars=None, cat_vars=None, num_algorithm='pca', cat_algorithm='mca', num_kwargs=None, cat_kwargs=None)
```

| Parameter | Type | Description |
|:-|:-|:-|
| `df` | `pandas.DataFrame` | Data table containing the data with the original variables |
| `num_vars` | `string`, `list`, `pandas.Series`, or `numpy.array` | Numerical variable name(s) |
| `cat_vars` | `string`, `list`, `pandas.Series`, or `numpy.array` | Categorical variable name(s) |
| `num_algorithm` | `string` | Algorithm to be used for dimensionality reduction of numerical variables. By default, PCA is used. The current version also supports SPCA |
| `cat_algorithm` | `string` | Algorithm to be used for dimensionality reduction of categorical variables. By default, MCA is used. The current version doesn’t support other algorithms |
| `num_kwargs` | `dictionary` | Additional keyword arguments to pass to the model used for numerical variables |
| `cat_kwargs` | `dictionary` | Additional keyword arguments to pass to the model used for categorical variables |
| **Attribute** | **Type** | **Description** |
| `n_components_` | `int` | Final number of extracted components |
| `min_explained_variance_ratio_` | `float` | Minimum explained variance ratio. By default, 0.5 |
| `num_trans_` | `pandas.DataFrame` | Extracted components from numerical variables |
| `cat_trans_` | `pandas.DataFrame` | Extracted components from categorical variables |
| `num_components_` | `list` | List of names assigned to the extracted components from numerical variables |
| `cat_components_` | `list` | List of names assigned to the extracted components from categorical variables |
| `pca_` | `sklearn.decomposition.PCA` | PCA instance used to speed up some computations and for comparison purposes |

<h4 id="DimensionalityReduction_class_methods">
Methods
</h4>

<h4>transform()</h4>

[Source](https://github.com/malgar/clust-learn/blob/f0744a15823c2b7c6b49c278d08d708d05df952a/clearn/dimensionality_reduction/dimensionality_reduction.py#L79)

```
transform(self, n_components=None, min_explained_variance_ratio=0.5)
```

Transforms a DataFrame df to a lower dimensional space.

<h4>num_main_contributors(()</h4>

[Source](https://github.com/malgar/clust-learn/blob/f0744a15823c2b7c6b49c278d08d708d05df952a/clearn/dimensionality_reduction/dimensionality_reduction.py#L191)

```
num_main_contributors(self, thres=0.5, n_contributors=None, dim_idx=None, component_description=None, col_description=None, output_path=None)
```

Computes the original numerical variables with the strongest relation to the derived variable(s) (measured as Pearson correlation coefficient).

<h4>cat_main_contributors(()</h4>

[Source](https://github.com/malgar/clust-learn/blob/f0744a15823c2b7c6b49c278d08d708d05df952a/clearn/dimensionality_reduction/dimensionality_reduction.py#L225)

```
cat_main_contributors(self, thres=0.14, n_contributors=None, dim_idx=None, component_description=None, col_description=None, output_path=None)
```

Computes the original categorical variables with the strongest relation to the derived variable(s)(measured as correlation ratio).

<h4>cat_main_contributors_stats()</h4>

[Source](https://github.com/malgar/clust-learn/blob/f0744a15823c2b7c6b49c278d08d708d05df952a/clearn/dimensionality_reduction/dimensionality_reduction.py#L259)

```
cat_main_contributors_stats(self, thres=0.14, n_contributors=None, dim_idx=None, output_path=None)
```

Computes for every categorical variable's value, the mean and std of the derived variables that are strongly related to the categorical variable (based on the correlation ratio)).

<h4>plot_num_explained_variance()</h4>

[Source](https://github.com/malgar/clust-learn/blob/f0744a15823c2b7c6b49c278d08d708d05df952a/clearn/dimensionality_reduction/dimensionality_reduction.py#L286)

```
plot_num_explained_variance(self, thres=0.5, plots='all', output_path=None, savefig_kws=None)
```

Plot the explained variance (ratio, cumulative, and/or normalized) for numerical variables.

<h4>plot_cat_explained_variance()</h4>

[Source](https://github.com/malgar/clust-learn/blob/f0744a15823c2b7c6b49c278d08d708d05df952a/clearn/dimensionality_reduction/dimensionality_reduction.py#L303)

```
plot_cat_explained_variance(self, thres=0.5, plots='all', output_path=None, savefig_kws=None)
```

Plot the explained variance (ratio, cumulative, and/or normalized) for categorical variables.

<h4>plot_num_main_contributors()</h4>

[Source](https://github.com/malgar/clust-learn/blob/f0744a15823c2b7c6b49c278d08d708d05df952a/clearn/dimensionality_reduction/dimensionality_reduction.py#L321)

```
plot_num_main_contributors(self, thres=0.5, n_contributors=5, dim_idx=None, output_path=None, savefig_kws=None)
```

Plot main contributors (original variables with the strongest relation with derived variables) for every derived variable.

<h4>plot_cat_main_contributor_distribution()</h4>

[Source](https://github.com/malgar/clust-learn/blob/f0744a15823c2b7c6b49c278d08d708d05df952a/clearn/dimensionality_reduction/dimensionality_reduction.py#L344)

```
plot_cat_main_contributor_distribution(self, thres=0.14, n_contributors=None, dim_idx=None, output_path=None, savefig_kws=None)
```

Plot main contributors (original variables with the strongest relation with derived variables) for every derived variable.

<h3 id="user-content-module-clustering">
7.iii. Clustering
</h3>

The `Clustering` class encapsulates all the functionality of this module and stores the data, the instances of the algorithms used, and other relevant information so it is always accessible.

<h4 id="Clustering_class">
Clustering class
</h4>

```
cl = Clustering(df, algorithms='kmeans', normalize=False)
```

| Parameter | Type | Description |
|:-|:-|:-|
| `df` | `pandas.DataFrame` | Data frame containing the data to be clustered |
| `algorithms` | `string` or `list` | Algorithms to be used for clustering. The current version supports k-means and agglomerative clustering |
| `normalize` | `bool` | Whether to apply data normalization for fair comparisons between variables. In case dimensionality reduction is applied beforehand, normalization should not be applied |
| **Attribute** | **Type** | **Description** |
| `dimensions_` | `list` | List of columns of they input data frame |
| `instances_` | `dict` | Pairs of algorithm name and its instance |
| `metric_` | `string` | The cluster validation metric used. Four metrics available: ['inertia', 'davies_bouldin_score', 'silhouette_score',  'calinski_harabasz_score'] |
| `optimal_config_` | `tuple` | Tuple with the optimal configuration for clustering containing the algorithm name, number of clusters, and value of the chosen validation metric |
| `scores_` | `dict` | Pairs of algorithm name and a list of values of the chosen validation metric for a cluster range |

<h4 id="Clustering_class_methods">
Methods
</h4>

<h4>compute_clusters()</h4>

[Source](https://github.com/malgar/clust-learn/blob/be4a2238670af01023bd419a0f8adaa7f9cee9f6/clearn/clustering/clustering.py#L117)

```
compute_clusters(self, n_clusters=None, metric='inertia', max_clusters=10, prefix=None, weights=None)
```

Calculates clusters.
If more than one algorithm is passed in the class constructor, first, the optimal number of clusters
is computed for each algorithm based on the metric passed to the method. Secondly, the algorithm that
provides the best performance for the corresponding optimal number of clusters is selected.
Therefore, the result shows the clusters calculated with the best performing algorithm based on the
criteria explained above.

<h4 id="describe_clusters">
describe_clusters()
</h4>

[Source](https://github.com/malgar/clust-learn/blob/be4a2238670af01023bd419a0f8adaa7f9cee9f6/clearn/clustering/clustering.py#L182)

```
describe_clusters(self, df_ext=None, variables=None, cluster_filter=None, statistics=['mean', 'median', 'std'], output_path=None)
```

Describes clusters based on internal or external *continuous* variables.
For categorical variables use [`describe_clusters_cat()`](#describe_clusters_cat).

<h4 id="describe_clusters_cat">
describe_clusters_cat()
</h4>

[Source](https://github.com/malgar/clust-learn/blob/be4a2238670af01023bd419a0f8adaa7f9cee9f6/clearn/clustering/clustering.py#L237)

```
describe_clusters_cat(self, cat_array, cat_name=None, order=None, normalize=False, output_path=None)
```

Describes clusters based on  external *categorical* variables. The result is a contingency table.
For continuous variables use [`describe_clusters()`](#describe_clusters).

<h4>compare_cluster_means_to_global_means()</h4>

[Source](https://github.com/malgar/clust-learn/blob/be4a2238670af01023bd419a0f8adaa7f9cee9f6/clearn/clustering/clustering.py#L276)

```
compare_cluster_means_to_global_means(self, df_original=None, output_path=None)
```

For every cluster and every internal variable, the relative difference between the intra-cluster mean
and the global mean.

<h4>anova_tests()</h4>

[Source](https://github.com/malgar/clust-learn/blob/be4a2238670af01023bd419a0f8adaa7f9cee9f6/clearn/clustering/clustering.py#L303)

```
anova_tests(self, df_test=None, vars_test=None, cluster_filter=None, output_path=None)
```

Runs ANOVA tests for a given set of continuous variables (internal or external) to test dependency with clusters.

<h4>chi2_test()</h4>

[Source](https://github.com/malgar/clust-learn/blob/be4a2238670af01023bd419a0f8adaa7f9cee9f6/clearn/clustering/clustering.py#L360)

```
chi2_test(self, cat_array)
```

Runs Chi-squared tests for a given categorical variable to test dependency with clusters.

<h4>plot_score_comparison()</h4>

[Source](https://github.com/malgar/clust-learn/blob/be4a2238670af01023bd419a0f8adaa7f9cee9f6/clearn/clustering/clustering.py#L379)

```
plot_score_comparison(self, output_path=None, savefig_kws=None)
```

Plots the comparison in performance between the different clustering algorithms.

<h4>plot_optimal_components_normalized()</h4>

[Source](https://github.com/malgar/clust-learn/blob/be4a2238670af01023bd419a0f8adaa7f9cee9f6/clearn/clustering/clustering.py#L400)

```
plot_optimal_components_normalized(self, output_path=None, savefig_kws=None)
```

Plots the normalized curve used for computing the optimal number of clusters.

<h4>plot_clustercount()</h4>

[Source](https://github.com/malgar/clust-learn/blob/be4a2238670af01023bd419a0f8adaa7f9cee9f6/clearn/clustering/clustering.py#L419)

```
plot_clustercount(self, output_path=None, savefig_kws=None)
```

Plots a bar plot with cluster counts.

<h4>plot_cluster_means_to_global_means_comparison()</h4>

[Source](https://github.com/malgar/clust-learn/blob/be4a2238670af01023bd419a0f8adaa7f9cee9f6/clearn/clustering/clustering.py#L432)

```
plot_cluster_means_to_global_means_comparison(self, df_original=None, xlabel=None, ylabel=None,
                                              levels=[-0.50, -0.32, -0.17, -0.05, 0.05, 0.17, 0.32, 0.50],
                                              output_path=None, savefig_kws=None)
```

Plots the normalized curve used for computing the optimal number of clusters.

<h4>plot_distribution_comparison_by_cluster()</h4>

[Source](https://github.com/malgar/clust-learn/blob/be4a2238670af01023bd419a0f8adaa7f9cee9f6/clearn/clustering/clustering.py#L469)

```
plot_distribution_comparison_by_cluster(self, df_ext=None, xlabel=None, ylabel=None, output_path=None, savefig_kws=None)
```

Plots the violin plots per cluster and *continuous* variables of interest to understand differences in their distributions by cluster.

<h4>plot_clusters_2D()</h4>

[Source](https://github.com/malgar/clust-learn/blob/be4a2238670af01023bd419a0f8adaa7f9cee9f6/clearn/clustering/clustering.py#L498)

```
plot_clusters_2D(self, coor1, coor2, style_kwargs=dict(), output_path=None, savefig_kws=None)
```

Plots two 2D plots:
	 - A scatter plot styled by the categorical variable `hue`.
	 - A 2D plot comparing cluster centroids and optionally the density area.
	 
<h4>plot_cat_distribution_by_cluster()</h4>

[Source](https://github.com/malgar/clust-learn/blob/be4a2238670af01023bd419a0f8adaa7f9cee9f6/clearn/clustering/clustering.py#L545)

```
plot_cat_distribution_by_cluster(self, cat_array, cat_label=None, cluster_label=None, output_path=None, savefig_kws=None)
```

Plots the relative contingency table of the clusters with a categorical variable as a stacked bar plot.

<h3 id="user-content-module-classifier">
7.iv. Classifier
</h3>

The functionality of this module is encapsulated in the `Classifier` class, which is also responsible for storing the original data, the instances of the models used, and any other relevant information.

<h4 id="Classifier_class">
Classifier class
</h4>

```
classifier = Classifier(df, predictor_cols, target, num_cols=None, cat_cols=None)
```

| Parameter | Type | Description |
|:-|:-|:-|
| `df` | `pandas.DataFrame` | Data frame containing the data |
| `predictor_cols` | `list` of `string` | List of columns to use as predictors |
| `target` | `numpy.array` or `list` | Values of the target variable |
| `num_cols` | `list` | List of numerical columns from predictor_cols |
| `cat_cols` | `list` | List of categorical columns from predictor_cols |
| **Attribute** | **Type** | **Description** |
| `filtered_features_` | `list` | List of columns of the input data frame |
| `model_` | Instance of `TransformerMixin` and `BaseEstimator` from `sklearn.base` | Trained classifier |
| `X_train_` | `numpy.array` | Train split of predictors |
| `X_test_` | `numpy.array` | Test split of predictors |
| `y_train_` | `numpy.array` | Train split of target |
| `y_test_` | `numpy.array` | Test split of target |
| `grid_result_` | `sklearn.model_selection.GridSearchCV` | Instance of fitted estimator for hyperparameter tuning |

<h4 id="Classifier_class_methods">
Methods
</h4>

<h4>train_model()</h4>

[Source](https://github.com/malgar/clust-learn/blob/5826ef273eb876c961eab7fa4eacb31caff25ef0/clearn/classifier/classifier.py#L52)

```
train_model(self, model=None, feature_selection=True, features_to_keep=[],
			feature_selection_model=None, hyperparameter_tuning=False, param_grid=None,
			train_size=0.8)
```

This method trains a classification model.

By default, it uses XGBoost, but any other estimator (instance of `scikit-learn.Estimator`) can be used.

The building process consists of three main steps:
 - Feature Selection (optional)
 
Feature removing highly correlated variables using a classification model and SHAP values
to determine which to keep, and Recursive Feature Elimination with Cross-Validation (RFECV)
on the remaining features.

 - Hyperparameter tuning (optional)
 
Runs grid search with cross-validation for hyperparameter tuning. **Note** the parameter grid
must be passed.

 - Model training
 
Trains a classification model with the selected features and hyperparameters. By default, an XGBoost
classifier will be trained.
   
**Note** both hyperparameter tuning and model training are run on a train set. Train-test split is performed
using `sklearn.model_selection.train_test_split`.

<h4>hyperparameter_tuning_metrics()</h4>

[Source](https://github.com/malgar/clust-learn/blob/5826ef273eb876c961eab7fa4eacb31caff25ef0/clearn/classifier/classifier.py#L134)

```
hyperparameter_tuning_metrics(self, output_path=None)
```

This method returns the average and standard deviation of the cross-validation runs for every hyperparameter
combination in hyperparameter tuning.

<h4>confusion_matrix()</h4>

[Source](https://github.com/malgar/clust-learn/blob/5826ef273eb876c961eab7fa4eacb31caff25ef0/clearn/classifier/classifier.py#L154)

```
confusion_matrix(self, test=True, sum_stats=True, output_path=None)
```

This method returns the confusion matrix of the classification model.

<h4>classification_report()</h4>

[Source](https://github.com/malgar/clust-learn/blob/5826ef273eb876c961eab7fa4eacb31caff25ef0/clearn/classifier/classifier.py#L195)

```
classification_report(self, test=True, output_path=None)
```

This method returns the `sklearn.metrics.classification_report` in `pandas.DataFrame` format.

This report contains the intra-class metrics precision, recall and F1-score, together with the global accuracy,
and macro average and weighted average of the three intra-class metrics.

<h4>plot_shap_importances()</h4>

[Source](https://github.com/malgar/clust-learn/blob/5826ef273eb876c961eab7fa4eacb31caff25ef0/clearn/classifier/classifier.py#L225)

```
plot_shap_importances(self, n_top=7, output_path=None, savefig_kws=None)
```

Plots shap importance values, calculated as the combined average of the absolute values of the shap values
for all classes.

<h4>plot_shap_importances_beeswarm()</h4>

[Source](https://github.com/malgar/clust-learn/blob/5826ef273eb876c961eab7fa4eacb31caff25ef0/clearn/classifier/classifier.py#L241)

```
plot_shap_importances_beeswarm(self, class_id, n_top=10, output_path=None, savefig_kws=None)
```

Plots a summary of shap values for a specific class of the target variable. This uses [shap beeswarm plot](https://shap.readthedocs.io/en/latest/example_notebooks/api_examples/plots/beeswarm.html).

<h4>plot_confusion_matrix()</h4>

[Source](https://github.com/malgar/clust-learn/blob/5826ef273eb876c961eab7fa4eacb31caff25ef0/clearn/classifier/classifier.py#L260)

```
plot_confusion_matrix(self, test=True, sum_stats=True, output_path=None, savefig_kws=None)
```

This function makes a pretty plot of an sklearn Confusion Matrix cf using a Seaborn heatmap visualization.

<h4>plot_roc_curves()</h4>

[Source](https://github.com/malgar/clust-learn/blob/5826ef273eb876c961eab7fa4eacb31caff25ef0/clearn/classifier/classifier.py#L280)

```
 plot_roc_curves(self, test=True, output_path=None, savefig_kws=None)
```

Plots ROC curve for every class.

<h2 id="user-content-citing">
8. Citing
</h2>

<<TO-DO>>


