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
	7.1. [Data processing](#user-content-module-preprocessing)
		7.1.1. [Data imputation](#user-content-module-preprocessing-imputation)
		7.1.2. [Outliers](#user-content-module-preprocessing-outliers)
	7.2. [Dimensionality reduction](#user-content-module-dimensionality)
	7.3. [Clustering](#user-content-module-clustering)
	7.4. [Classifier](#user-content-module-classifier)
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
7.1. Data preprocessing
</h3>

Data preprocessing consists of a set of manipulation and transformation tasks performed on the raw data before it is used for its analysis. Although data quality is essential for obtaining robust and reliable results, real-world data is often incomplete, noisy, or inconsistent. Therefore, data preprocessing is a crucial step in any analytical study.

<h4 id="user-content-module-preprocessing-imputation">
7.1.1. Data imputation
</h4>

<h5 id="impute_missing_values">
`impute_missing_values`
</h5>

```
impute_missing_values(df, num_vars, cat_vars, num_pair_kws=None, mixed_pair_kws=None, cat_pair_kws=None, graph_thres=0.05, k=8, max_missing_thres=0.33)
```

This function imputes missing values following this steps:
1. One-to-one model based imputation for strongly related variables.
2. Cluster based hot deck imputation where clusters are obtained as the connected components of an undirected graph *G=(V,E)*, where *V* is the set of variables and *E* the pairs of variables with mutual information above a predefined threshold.
3. Records with a proportion of missing values above a predefined threshold are discarded to ensure the quality of the hot deck imputation.
4. Hot deck imputation for the remaining missing values considering all variables together.


<h4 id="user-content-module-preprocessing-outliers">
7.1.2. Outliers
</h4>

<<TO-DO>>

<h3 id="user-content-module-dimensionality">
7.2. Dimensionality reduction
</h3>

<<TO-DO>>

<h3 id="user-content-module-clustering">
7.3. Clustering
</h3>

<<TO-DO>>

<h3 id="user-content-module-classifier">
7.4. Classifier
</h3>

<<TO-DO>>

<h2 id="user-content-citing">
8. Citing
</h2>

<<TO-DO>>


