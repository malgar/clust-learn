"""Utils for the classification module"""
# Author: Miguel Alvarez-Garcia

import logging
import numpy as np
import pandas as pd
import shap

from sklearn.feature_selection import RFECV
from sklearn.model_selection import GridSearchCV
from ..utils import compute_high_corr_pairs, compute_highly_related_categorical_vars, compute_highly_related_mixed_vars

logging.basicConfig(
    format='%(asctime)s %(levelname)-8s %(name)s: %(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S')

logger = logging.getLogger(__name__)


def get_shap_importances(classifier, X):
    """
    Computes shap importance values as the combined average of the absolute values of the shap values
    for all classes.

    Parameters
    ----------
    classifier : object
        Classification model supported by SHAP (e.g., sklearn models or xgboost).
    X : `numpy.ndarray` or `pandas.DataFrame`
        Observations.

    Returns
    ----------
    importances : `pandas.DataFrame`
        DataFrame with predictors and their corresponding importance.
    """
    explainer = shap.Explainer(classifier)
    shap_values = explainer(X)
    shap_importance_values = [np.abs(shap_values.values[:, i, :]).mean() for i in range(shap_values.values.shape[1])]
    importances = pd.DataFrame(
        data={'variable_name': shap_values.feature_names, 'shap_importance': shap_importance_values}).sort_values(
        'shap_importance', ascending=False).reset_index(drop=True)
    return importances


def _compute_highly_related_pairs(df, num_vars=None, cat_vars=None, num_kws=None, mixed_kws=None, cat_kws=None):
    """
    Computes strongly related pairs of variables. Depending on the type of variables, a correlation coefficient
    (numerical variables), partial eta squared (mixed-type variables), or mutual information (categorical variables) is
    used to measured the strength of the relationship of each pair.

    Parameters
    ----------
    df : `pandas.DataFrame`
        DataFrame containing the data.
    num_vars : str, list, series, or vector array
        Numerical variable name(s).
    cat_vars : str, list, series, or vector array
        Categorical variable name(s).
    {num,mixed,cat}_kws : dict, default=None
        Additional keyword arguments to pass to `compute_high_corr_pairs()`, `compute_highly_related_mixed_vars()`, and
        `compute_highly_related_categorical_vars()`.

    Returns
    ----------
    final_pairs : `pandas.DataFrame`
        DataFrame with pairs of highly correlated variables (var1: variable with values to impute; var2: variable to be
        used as independent variable for model-based imputation).
    """
    if num_vars is None and cat_vars is None:
        raise ValueError('Numerical or categorical variable lists are required.')

    # Numerical variable pairs (correlation)
    num_pairs = pd.DataFrame()
    if num_vars:
        num_kws = num_kws if num_kws else dict()
        num_pairs = compute_high_corr_pairs(df[num_vars], **num_kws)

    # Mixed variable pairs (cross correlation)
    mixed_pairs = pd.DataFrame()
    if num_vars and cat_vars:
        mixed_kws = mixed_kws if mixed_kws else dict()
        mixed_pairs = compute_highly_related_mixed_vars(df, num_vars, cat_vars, **mixed_kws)

    # Categorical variable pairs (mutual information)
    cat_pairs = pd.DataFrame()
    if cat_vars:
        cat_kws = cat_kws if cat_kws else dict()
        cat_pairs = compute_highly_related_categorical_vars(df[cat_vars], **cat_kws)

    final_pairs = pd.concat([num_pairs, mixed_pairs, cat_pairs], ignore_index=True)
    return final_pairs


def run_feature_selection(df, original_features, target, classifier, num_vars=None, cat_vars=None, features_to_keep=[],
                          num_kws=None, mixed_kws=None, cat_kws=None, rfecv_kws=None):
    """
    Performs feature selection in three steps:
        - First, if some features must be kept (informed in `features_to_keep`), those other
          features that highly related with those in `features_to_keep` are removed.

        - Next, a classifier is iteratively trained to obtain feature shap importances.
          In each iteration, the feature with the highest shape importance which has not been previously
          visited is selected and all other highly related features are removed.

        - Finally, Recursive Feature Elimination with Cross-Validation (RFECV) is applied on the remaining
          features.

    Parameters
    ----------
    df : `pandas.DataFrame`
        DataFrame containing, at least, all variables in `original_features`
    original_features : `numpy.array` or list
        Array of original features to select from. `original_features` = `num_vars` + `cat_vars`.
    target : `numpy.array` or list
        Array with target values. This is the dependent variable values.
    classifier : estimator
        Should be an instance of `scikit-learn.Estimator`, `scikit-learn.Predictor`, `scikit-learn.Transformer`,
        and `scikit-learn.Model`.
    num_vars : `numpy.array` or list, default=None
        Array of numerical features. All variable in `num_vars` must be in `original_features`.
    cat_vars : `numpy.array` or list, default=None
        Array of categorical features. All variable in `cat_vars` must be in `original_features`.
    features_to_keep: list, default=[]
        In case some features are of special interest to the analysis and should be kept.
    {num,mixed,cat}_kws : dict, default=None
        Additional keyword arguments to pass to `compute_high_corr_pairs()`, `compute_highly_related_mixed_vars()`, and
        `compute_highly_related_categorical_vars()`.
    rfecv_kws : dictionary, default=None
        Dictionary for RFECV.

    Returns
    ----------
    filtered_features : list
        List with selected features.
    """

    tot_vars = 0
    if cat_vars:
        tot_vars += len(cat_vars)
    if num_vars:
        tot_vars += len(num_vars)
    assert len(original_features) == tot_vars, "`original_features` != `num_vars` + `cat_vars`"

    # First, we compute highly related pairs of variables
    if num_kws is None:
        num_kws = dict(corr_thres=0.8)
    hi_rel = _compute_highly_related_pairs(df, num_vars, cat_vars, num_kws, mixed_kws, cat_kws)

    filtered_features = original_features.copy()

    if hi_rel.shape[0] > 0:
        # Next, we remove variables highly correlated with the ones indicated to be kept
        for v in features_to_keep:
            for v2 in hi_rel.loc[hi_rel['var1'] == v, 'var2'].to_list():
                if v2 in filtered_features:
                    if v2 in features_to_keep:
                        logger.warning(
                            f'Variables {v} and {v2} are highly correlated, and both were selected to be kept.')
                    else:
                        filtered_features.remove(v2)

        # We update the highly-correlated pair DataFrame for efficiency purposes
        hi_rel = _compute_highly_related_pairs(df, num_vars, cat_vars, num_kws, mixed_kws, cat_kws)

    if hi_rel.shape[0] > 0:
        # Next, we remove highly correlated variables iteratively, keeping the most impactful one
        # We identify the most impactful one using a classification algorithm and SHAP importances
        # In every iteration, we remove all variables with a high correlation to the unvisited variable
        # with the highest SHAP importance value
        stop = False
        visited = []
        while not stop:
            X = df[filtered_features]
            y = target
            classifier.fit(X, y)
            importances = get_shap_importances(classifier, X)
            updated = False
            for feat in importances['variable_name']:
                if feat in hi_rel['var1'].to_list() and feat not in visited:
                    visited.append(feat)
                    for var2 in hi_rel.loc[hi_rel['var1'] == feat, 'var2'].to_list():
                        if var2 in filtered_features and var2 not in features_to_keep:
                            filtered_features.remove(var2)
                    updated = True
                    break

            stop = not updated

    # Finally, we apply Recursive Feature Elimination with Cross-Validation (RFECV)
    X = df[filtered_features]
    y = target
    if rfecv_kws is None:
        rfecv_kws = dict(step=1, cv=3)
    selector = RFECV(classifier, **rfecv_kws)
    selector = selector.fit(X, y)
    filtered_features = list(np.array(filtered_features)[np.where(selector.support_)])
    return filtered_features


def run_hyperparameter_tuning(X_train, y_train, classifier, param_grid, gridsearch_kws=None):
    """
    Runs grid search with cross-validation for hyperparameter tuning.

    Parameters
    ----------
    X_train : `np.ndarray` or `pandas.DataFrame`
        Training set.
    y_train : `numpy.array` or list
        Training target values.
    classifier : estimator, default=None
        Should be an instance of `scikit-learn.Estimator`, `scikit-learn.Predictor`, `scikit-learn.Transformer`,
        and `scikit-learn.Model`.
        If none is passed a `sklearn.ensemble.RandomForestClassifier` model is used.
    param_grid : dictionary
        Dictionary with the different hyperparameters and their corresponding values to be evaluated.
    gridsearch_kws : dictionary, default=None
        Dictionary for grid search.

    Returns
    ----------
    grid_result : `sklearn.model_selection.GridSearchCV`
        Instance of fitted estimator.
    """
    if gridsearch_kws is None:
        gridsearch_kws = dict(n_jobs=-1, cv=3)
    grid_search = GridSearchCV(classifier, param_grid, **gridsearch_kws)
    grid_result = grid_search.fit(X_train, y_train)
    return grid_result
