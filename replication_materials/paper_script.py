# This script contains the code used for the illustration example

import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd

# Set the seed
np.random.seed(42)

# Numerical and categorical variable lists
num_vars = ['AGE', 'PAREDINT', 'BMMJ1',
            'BFMJ2', 'HISEI', 'DURECEC', 'BSMJ', 'MMINS',
            'LMINS', 'SMINS', 'TMINS', 'FCFMLRTY', 'SCCHANGE', 'CHANGE', 'STUBMI',
            'ESCS', 'UNDREM', 'METASUM', 'METASPAM', 'ICTHOME', 'ICTSCH', 'HOMEPOS',
            'CULTPOSS', 'HEDRES', 'WEALTH', 'ICTRES', 'DISCLIMA', 'TEACHSUP',
            'DIRINS', 'PERFEED', 'EMOSUPS', 'STIMREAD', 'ADAPTIVITY', 'TEACHINT',
            'JOYREAD', 'SCREADCOMP', 'SCREADDIFF', 'PERCOMP', 'PERCOOP', 'ATTLNACT',
            'COMPETE', 'WORKMAST', 'GFOFAIL', 'EUDMO', 'SWBP', 'RESILIENCE',
            'MASTGOAL', 'GCSELFEFF', 'GCAWARE', 'ATTIMM', 'INTCULT', 'PERSPECT',
            'COGFLEX', 'RESPECT', 'AWACOM', 'GLOBMIND', 'DISCRIM', 'BELONG',
            'BEINGBULLIED', 'ENTUSE', 'HOMESCH', 'USESCH', 'INTICT', 'COMPICT',
            'AUTICT', 'SOIAICT', 'ICTCLASS', 'ICTOUTSIDE', 'INFOCAR', 'INFOJOB1',
            'INFOJOB2', 'FLCONFIN', 'FLCONICT', 'FLSCHOOL', 'FLFAMILY', 'BODYIMA',
            'SOCONPA']
cat_vars = ['ST004D01T', 'IMMIG', 'REPEAT']

# Load data
df = pd.read_csv('data/pisa_spain_sample_v2.csv')

# DATA PREPROCESSING
print('--- DATA PREPROCESSING ---')
from clearn.data_preprocessing import *

# Computre missing values
n_missing = df.isnull().sum().sum()
print('Missing values:', n_missing, f'({n_missing*100/df.size}%)')

# Generate missing values heat map
missing_values_heatmap(df, output_path=os.path.join("img", "missing_heatmap.jpg"))

# Impute missing values
df_imp = impute_missing_values(df, num_vars=num_vars, cat_vars=cat_vars)

# Plot the comparison between the distribution of a selection of variables before and after imputation
plot_imputation_distribution_assessment(df.loc[df_imp.index], df_imp, ['COMPICT', 'BODYIMA', 'PERCOOP', 'SOCONPA'],
                                        output_path=os.path.join("img", "imputation_distribution_assessment.jpg"))

# Remove outliers
df_, outliers = remove_outliers(df_imp, num_vars+cat_vars)

# DIMENSIONALITY REDUCTION
print('--- DIMENSIONALITY REDUCTION ---')
df_ = df_.reset_index(drop=True)
from clearn.dimensionality_reduction import DimensionalityReduction

# Instantiate class, project to a lower dimensionality with optimal number of components
dr = DimensionalityReduction(df_, num_vars=num_vars, cat_vars=cat_vars, num_algorithm='spca')
df_t = dr.transform(min_explained_variance_ratio=None)
print(dr.n_components_, len(dr.num_components_), len(dr.cat_components_))

# Explain fourth extracted component
print(dr.num_main_contributors(dim_idx=3))
dr.plot_num_main_contributors(dim_idx=3, output_path=os.path.join("img", "dim_red_main_contributors.jpg"))

# Explain component extracted from categorical variables
print(dr.cat_main_contributors_stats())
dr.plot_cat_main_contributor_distribution(dim_idx=0, output_path=os.path.join("img", "dim_red_cat_component.jpg"))

# Explained variance + elbow method
dr.plot_num_explained_variance(0.5, plots=['cumulative', 'normalized'],
                               output_path=os.path.join("img", "dim_red_explained_variance.jpg"))

# CLUSTERING
print('--- CLUSTERING ---')
from clearn.clustering import Clustering

# Instantiate class and compute clusters on projected space
cl = Clustering(df_t, algorithms=['kmeans', 'ward'], normalize=False)
cl.compute_clusters(max_clusters=21, prefix='STU')
print(cl.optimal_config_)

# Plot number of observations per cluster
cl.plot_clustercount(output_path=os.path.join("img", "cluster_count.jpg"))

# Plot normalized WSS curve for optimal k selection
cl.plot_optimal_components_normalized(output_path=os.path.join("img", "clustering_elbow_curve.jpg"))

# Compare intra-cluster means to global means
print(cl.compare_cluster_means_to_global_means())
cl.plot_cluster_means_to_global_means_comparison(xlabel='Principal Components', ylabel='Clusters',
                                                 levels=[-1, -0.67, -0.3, -0.15, 0.15, 0.3, 0.67, 1],
                                                 output_path=os.path.join("img", "clustering_intra_comparison.jpg"))

# Compare distributions of original variables by cluster
cl.plot_distribution_comparison_by_cluster(df_ext=df_[['ESCS', 'TEACHSUP']],
                                           output_path=os.path.join("img", "clustering_distribution_comparison.jpg"))

# 2-D plots
cl.plot_clusters_2D('dim_01', 'dim_02', output_path=os.path.join("img", "clustering_2d_plots.jpg"))

# Comparison of original categorical variable distribution by cluster
print(cl.describe_clusters_cat(df_['IMMIG'], cat_name='IMMIG', normalize=True))
cl.plot_cat_distribution_by_cluster(df_['IMMIG'], cat_label='IMMIG', cluster_label='Student clusters',
                                    output_path=os.path.join("img", "clustering_cat_comparison.jpg"))

# Assign clusters to data frame with original variables
df_['cluster'] = cl.df['cluster'].values
df_['cluster_cat'] = cl.df['cluster_cat'].values

# CLASSIFIER
print('--- CLASSIFIER ---')
from clearn.classifier import Classifier
np.random.seed(42)

# Instantiate the class with the original variables. As target, we set the clusters computed above
var_list = list(df_.columns[1:-3])
classifier = Classifier(df_, predictor_cols=var_list, target=df_['cluster'], num_cols=num_vars, cat_cols=cat_vars)

# Build a pipeline with feature selection, hyperparameter tuning, and model fitting. For feature selection, we make sure
# that variable ESCS (economic, social and cultural status) gets selected (features_to_keep=['ESCS']). Hyperparameter
# optimization is performed through exhaustive grid search for different values of the number of estimators
# (n_estimators=[30, 60]), eta (eta=[0.15, 0.25]), and maximum tree depth (max_depth=[3, 5, 7]). The algorithms used are
# those configured by default, i.e. random forest for feature selection and xgboost for the final classification model.
classifier.train_model(features_to_keep=['ESCS'], hyperparameter_tuning=True,
                       param_grid=dict(n_estimators=[30, 60], eta=[0.15, 0.25], max_depth=[3, 5, 7]))

# Get global feature importance of the five most important features
print(classifier.feature_importances.head())

# Plot global feature importance
classifier.plot_shap_importances(output_path=os.path.join("img", "classifier_global_feature_importance.jpg"))
plt.clf()

# Plot local importance for clusters with identifiers 1 and 2
classifier.plot_shap_importances_beeswarm(class_id=1,
                                          output_path=os.path.join("img", "classifier_local_importance_cl1.jpg"))
plt.clf()
classifier.plot_shap_importances_beeswarm(class_id=2,
                                          output_path=os.path.join("img", "classifier_local_importance_cl2.jpg"))
plt.clf()

# Performance assessment
# Hyperparameter tuning results
print(classifier.hyperparameter_tuning_metrics())

# Confusion matrix
classifier.plot_confusion_matrix(output_path=os.path.join("img", "classifier_confusion_matrix.jpg"))

# Classification report
print(classifier.classification_report())

# ROC curves
classifier.plot_roc_curves(output_path=os.path.join("img", "classifier_roc_curves.jpg"))







