<!-- title -->
<div align="center">
  <h2>ldbx: Large Database Explainer</h2>
</div>

<!-- Short description -->
<p align="center">
   A Python package for extracting valuable information from large databases.
</p>

<hr>

## Table of contents
- [1. Introduction](#introduction)
- [[#user-content-intro][1 Introduction]]
- [[#user-content-architecture][2 Overall architecture]]
- [[#user-content-implementation][3 Implementation]]
- [[#user-content-install][4 Installation and configuration]]
- [[#user-content-license][5 Version and license information]]
- [[#user-content-future][6 Bug reports and future work]]
- [[#user-content-api][7 User guide & API]]
	- [[#user-content-module-preprocessing][7.1 Data preprocessing]]
		- [[#user-content-module-preprocessing-imputation][7.1.1 Data imputation]
		- [[#user-content-module-preprocessing-outliers][7.1.2 Outliers]
	- [[#user-content-module-dimensionality][7.2 Dimensionality reduction]]
	- [[#user-content-module-clustering][7.3 Clustering]]
	- [[#user-content-module-classifier][7.4 Classifier]]

<h2 id="introduction">
1. Introduction
</h2>

Here the intro

@@html:<a name="intro">@@
* 1 Introduction
:PROPERTIES:
:CUSTOM_ID: user-content-intro
:END:
<<TO-DO>>

@@html:<a name="architecture">@@
* 2 Overall architecture
:PROPERTIES:
:CUSTOM_ID: user-content-architecture
:END:
<<TO-DO>>

@@html:<a name="implementation">@@
* 3 Implementation
:PROPERTIES:
:CUSTOM_ID: user-content-implementation
:END:
<<TO-DO>>

@@html:<a name="install">@@
* 4 Installation and configuration
:PROPERTIES:
:CUSTOM_ID: user-content-install
:END:
<<TO-DO>>

@@html:<a name="license">@@
* 5 Version and license information
:PROPERTIES:
:CUSTOM_ID: user-content-license
:END:
<<TO-DO>>

@@html:<a name="future">@@
* 6 Bug reports and future work
:PROPERTIES:
:CUSTOM_ID: user-content-future
:END:
<<TO-DO>>

@@html:<a name="api">@@
* 7 User guide & API
:PROPERTIES:
:CUSTOM_ID: user-content-api
:END:

`ldbx` is organized into four modules:
	1. Data preprocessing
	2. Dimensionality reduction
	3. Clustering
	4. Classifier

Figure n shows the package layout with the functionalities covered by each module along with the techniques used, the explainability strategies available, and the main functions and class methods encapsulating these techniques and explainability strategies.

The four modules are designed to be used sequentially to ensure robust and explainable results. However, each of them is independent and can be used separately to suit different use cases.

@@html:<a name="module-preprocessing">@@
** 7.1 Data preprocessing
:PROPERTIES:
:CUSTOM_ID: user-content-module-preprocessing
:END:

Data preprocessing consists of a set of manipulation and transformation tasks performed on the raw data before it is used for its analysis. Although data quality is essential for obtaining robust and reliable results, real-world data is often incomplete, noisy, or inconsistent. Therefore, data preprocessing is a crucial step in any analytical study.

@@html:<a name="module-preprocessing-imputation">@@
*** 7.1.1 Data imputation
:PROPERTIES:
:CUSTOM_ID: user-content-module-preprocessing-imputation
:END:



@@html:<a name="module-preprocessing-outliers">@@
*** 7.1.2 Outliers
:PROPERTIES:
:CUSTOM_ID: user-content-module-preprocessing-outliers
:END:



@@html:<a name="module-dimensionality">@@
** 7.2 Dimensionality reduction
:PROPERTIES:
:CUSTOM_ID: user-content-module-dimensionality
:END:



@@html:<a name="module-clustering">@@
** 7.3 Clustering
:PROPERTIES:
:CUSTOM_ID: user-content-module-clustering
:END:




@@html:<a name="module-classifier">@@
** 7.4 Classifier
:PROPERTIES:
:CUSTOM_ID: user-content-module-classifier
:END:

