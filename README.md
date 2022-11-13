<!-- title -->
<div align="center">
  <h3>ldbx: Large Database Explainer</h3>
</div>

<!-- Short description -->
<p align="center">
   A Python package for extracting valuable information from large databases.
</p>

<hr>

## Table of contents
1. [Introduction](#user-content-introduction)
2. [Overall architecture](#user-content-architecture)
3. [Implementation](#user-content-implementation)
4. [Installation and configuration](#user-content-install)
5. [Version and license information](#user-content-license)
6. [Bug reports and future work](#user-content-future)
7. [User guide & API](#user-content-api)
	1. [Data processing](#user-content-module-preprocessing)
		1. [Data imputation](#user-content-module-preprocessing-imputation)
			- [compute_missing()](#compute_missing)
			- [impute_missing_values()](#impute_missing_values)
			- [missing_values_heatmap()](#missing_values_heatmap)
		2. [Outliers](#user-content-module-preprocessing-outliers)
			- [remove_outliers()](#remove_outliers)
	2. [Dimensionality reduction](#user-content-module-dimensionality)
		- [DimensionalityReduction class](#DimensionalityReduction_class)
	3. [Clustering](#user-content-module-clustering)
		- [Clustering class](#Clustering_class)
	4. [Classifier](#user-content-module-classifier)
		- [Classifier class](#Classifier_class)
8. [Citing](#user-content-citing)

<h2 id="user-content-introduction">
1. Introduction
</h2>

<<TO-DO>>

<h2 id="user-content-architecture">
2. Overall architecture
</h2>

<<TO-DO>>

<h2 id="user-content-implementation">
3. Implementation
</h2>

<<TO-DO>>

<h2 id="user-content-install">
4. Installation and configuration
</h2>

<<TO-DO>>

<h2 id="user-content-license">
5. Version and license information
</h2>

<<TO-DO>>

<h2 id="user-content-future">
6. Bug reports and future work
</h2>

<<TO-DO>>

<h2 id="user-content-api">
7. User guide & API
</h2>

`ldbx` is organized into four modules:
	1. Data preprocessing
	2. Dimensionality reduction
	3. Clustering
	4. Classifier

Figure n shows the package layout with the functionalities covered by each module along with the techniques used, the explainability strategies available, and the main functions and class methods encapsulating these techniques and explainability strategies.

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
	- DataFrame with pairs of highly correlated variables (var1: variable with values to impute; var2: variable to be used as independent variable for model-based imputation), together proportion of missing values of variables var1 and var2.

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
|-|-|-|
| `df` | `pandas.DataFrame` | Data frame containing the data to be clustered |
| `algorithms` | `string` or `list` | Algorithms to be used for clustering. The current version supports k-means and agglomerative clustering |
| `normalize` | `bool` | Whether to apply data normalization for fair comparisons between variables. In case dimensionality reduction is applied beforehand, normalization should not be applied |
| **Attribute** | **Type** | **Description** |
| `dimensions_` | `list` | List of columns of they input data frame |
| `instances_` | `dict` | Pairs of algorithm name and its instance |
| `metric_` | `string` | The cluster validation metric used. Four metrics available: ['inertia', 'davies_bouldin_score', 'silhouette_score',  'calinski_harabasz_score'] |
| `optimal_config_` | `tuple` | Tuple with the optimal configuration for clustering containing the algorithm name, number of clusters, and value of the chosen validation metric |
| `scores_` | `dict` | Pairs of algorithm name and a list of values of the chosen validation metric for a cluster range |


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
|-|-|-|
| df | pandas.DataFrame | Data frame containing the data |
| predictor_cols | List of string | List of columns to use as predictors |
| target | numpy.array or list | Values of the target variable |
| num_cols | list | List of numerical columns from predictor_cols |
| cat_cols | list | List of categorical columns from predictor_cols |
| **Attribute** | **Type** | **Description** |
| filtered_features_ | list | List of columns of the input data frame |
| model_ | Instance of TransformerMixin and BaseEstimator from sklearn.base | Trained classifier |
| X_train_ | numpy.array | Train split of predictors |
| X_test_ | numpy.array | Test split of predictors |
| y_train_ | numpy.array | Train split of target |
| y_test_ | numpy.array | Test split of target |
| grid_result_ | sklearn.model_selection.GridSearchCV | Instance of fitted estimator for hyperparameter tuning |

<h2 id="user-content-citing">
8. Citing
</h2>

<<TO-DO>>


