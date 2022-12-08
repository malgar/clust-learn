"""Classification class"""
# Author: Miguel Alvarez-Garcia

import logging
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

from .utils import run_feature_selection, run_hyperparameter_tuning, get_shap_importances
from .viz_utils import *


class Classifier:
    """
    Class to manage the classification model.

    Parameters
    ----------
    df : `pandas:DatasFrame`
        DataFrame with main data
    predictor_cols : list
        List of columns to use as predictors.
    target : `numpy.array` or list
        values of the target variable.
    num_cols : list, default=None
        List of numerical columns to use as predictors.
    cat_cols : list, default=None
        List of categorical columns to use as predictors.
    """

    def __init__(self, df, predictor_cols, target, num_cols=None, cat_cols=None):
        self.df = df
        self.original_features = predictor_cols
        self.num_vars = num_cols
        self.cat_vars = cat_cols
        self.filtered_features_ = None
        self.target = target
        self.model_ = None
        self.X_train_, self.X_test_, self.y_train_, self.y_test_ = tuple([None]*4)
        self.grid_result_ = None

        # Initialize logger
        logging.basicConfig(format='%(asctime)s %(levelname)-8s %(name)s: %(message)s',
                            level=logging.INFO,
                            datefmt='%Y-%m-%d %H:%M:%S')

        self.logger = logging.getLogger(__name__)

    def train_model(self, model=None, feature_selection=True, features_to_keep=[],
                    feature_selection_model=None, hyperparameter_tuning=False, param_grid=None,
                    train_size=0.8):
        """
        This method trains a classification model.

        By default, it uses XGBoost, but any other estimator (instance of `scikit-learn.Estimator`) can be used.

        The building process consists of four steps:
         - Train-test split
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

        Parameters
        ----------
        model : `scikit-learn.Estimator`, default=None
            Model to use as classifier. By default, `xgboost.XGBClassifier`.
        feature_selection : boolean, default=True
            If True, feature selection is performed on the original features.
        features_to_keep : list, default=empty list
            Features to be kept during the feature selection process.
        feature_selection_model : `scikit-learn.Estimator`, default=None
            Model to be used for feature selection. By default, `sklearn.ensemble.RandomForestClassifier`.
        hyperparameter_tuning : boolean, default=False
            If True, hyperparameter tuning is run using the parameter grid `param_grid`.
        param_grid : dictionary, default=None
            Parameter grid for hyperparameter tuning. **Note** if hyperparameter_tuning=True, a parameter grid is
            needed.
        train_size : float, default=0.8
            Size of the train split of the data for model training.
        """
        # Data is split into train and test sets
        X = self.df[self.original_features]
        y = self.target
        self.X_train_, self.X_test_, self.y_train_, self.y_test_ = train_test_split(X, y, train_size=train_size)

        # Feature selection
        if feature_selection:
            self.logger.info('Running feature selection...')
            if feature_selection_model is None:
                min_samples_leaf = int(np.ceil(self.X_train_.shape[0] * 0.05))
                feature_selection_model = RandomForestClassifier(max_depth=10,  min_samples_leaf=min_samples_leaf,
                                                                 random_state=42)

            self.filtered_features_ = run_feature_selection(self.df.loc[self.X_train_.index], self.original_features,
                                                            self.target.loc[self.X_train_.index],
                                                            feature_selection_model, self.num_vars, self.cat_vars,
                                                            features_to_keep)

            self.X_train_ = self.X_train_[self.filtered_features_]
            self.X_test_ = self.X_test_[self.filtered_features_]
        else:
            self.filtered_features_ = self.original_features.copy()

        # Model instantiation
        if model is None:
            model = XGBClassifier(eval_metric='auc', use_label_encoder=False, random_state=42)
        self.model_ = model

        # Hyperparameter tuning
        if hyperparameter_tuning:
            if param_grid is None:
                raise RuntimeError('For hyperparameter tuning, some parameter grid must be passed - `param_grid`')
            self.logger.info('Running hyperparameter tuning...')
            self.grid_result_ = run_hyperparameter_tuning(self.X_train_, self.y_train_, self.model_, param_grid)
            self.model_.set_params(**self.grid_result_.best_params_)

        # Model training
        self.logger.info('Training model...')
        self.model_.fit(self.X_train_, self.y_train_)
        self.logger.info('DONE!')

    @property
    def feature_importances(self):
        return get_shap_importances(self.model_, self.X_train_)

    def hyperparameter_tuning_metrics(self, output_path=None):
        """
        This method returns the average and standard deviation of the cross-validation runs for every hyperparameter
        combination in hyperparameter tuning.

        Parameters
        ----------
        output_path : str, default=None
            If an output_path is passed, the resulting DataFame is saved as a CSV file.
        """
        htm = pd.DataFrame(self.grid_result_.cv_results_['params'])
        htm.columns = pd.MultiIndex.from_product([[f'{self.model_.__class__.__name__} Hyperparameters'], htm.columns])
        htm[('Performance metrics', 'mean_test_score')] = self.grid_result_.cv_results_['mean_test_score']
        htm[('Performance metrics', 'std_test_score')] = self.grid_result_.cv_results_['std_test_score']

        if output_path is not None:
            htm.to_csv(output_path, index=False)

        return htm

    def confusion_matrix(self, test=True, sum_stats=True, output_path=None):
        """
        This method returns the confusion matrix of the classification model.

        Parameters
        ----------
        test : boolean, default=True
            If True, returns the confusion matrix calculated on the test set. If False, returns the confusion matrix on
            the train set.
        sum_stats : boolean, default=True
            If True, adds row (recall), column (precision), and global accuracy metrics to the matrix.
        output_path : str, default=None
            If an output_path is passed, the resulting DataFame is saved as a CSV file.

        Returns
        ----------
        cm : `pandas.DataFrame`
            DataFrame with confusion matrix.
        """
        X = self.X_test_ if test else self.X_train_
        y = self.y_test_ if test else self.y_train_

        cm = pd.DataFrame(confusion_matrix(y, self.model_.predict(X)),
                          columns=pd.MultiIndex.from_product([['Predicted values'], np.unique(self.target)]),
                          index=pd.MultiIndex.from_product([['Observed values'], np.unique(self.target)]))

        if sum_stats:
            # Precision, recall, and global accuracy are appended to the table
            recall = np.diag(cm) / cm.sum(1)
            precision = np.diag(cm) / cm.sum()
            accuracy = np.diag(cm).sum() / cm.sum().sum()
            cm['recall'] = recall
            cm = cm.transpose()
            cm['precision'] = list(precision) + [accuracy]
            cm = cm.transpose()

        if output_path is not None:
            cm.to_csv(output_path, index=False)

        return cm

    def classification_report(self, test=True, output_path=None):
        """
        This method returns the `sklearn.metrics.classification_report` in `pandas.DataFrame` format.
        This report contains the intra-class metrics precision, recall and F1-score, together with the global accuracy,
        and macro average and weighted average of the three intra-class metrics.

        Parameters
        ----------
        test : boolean, default=True
            If True, returns the confusion matrix calculated on the test set. If False, returns the confusion matrix on
            the train set.
        output_path : str, default=None
            If an output_path is passed, the resulting DataFame is saved as a CSV file.

        Returns
        ----------
        report : `pandas.DataFrame`
            DataFrame with classification report.
        """
        X = self.X_test_ if test else self.X_train_
        y = self.y_test_ if test else self.y_train_

        report = pd.DataFrame(classification_report(y, self.model_.predict(X), output_dict=True)).transpose()

        if output_path is not None:
            # In this case we do want to keep the index
            report.to_csv(output_path)

        return report

    def plot_shap_importances(self, n_top=7, output_path=None, savefig_kws=None):
        """
        Plots shap importance values, calculated as the combined average of the absolute values of the shap values
        for all classes.

        Parameters
        ----------
        n_top : int, default=7
           Top n features to be displayed. The importance of the rest are aggregated and displayed under the tag "Rest".
        output_path : str, default=None
           Path to save figure as image.
        savefig_kws : dict, default=None
           Save figure options.
        """
        plot_shap_importances(self.model_, self.X_train_, n_top=n_top, output_path=output_path, savefig_kws=savefig_kws)

    def plot_shap_importances_beeswarm(self, class_id, n_top=10, output_path=None, savefig_kws=None):
        """
        Plots a summary of shap values for a specific class of the target variable. This uses shap beeswarm plot
        (https://shap.readthedocs.io/en/latest/example_notebooks/api_examples/plots/beeswarm.html).

        Parameters
        ----------
        class_id : int
            The class for which to show the SHAP values.
        n_top : int, default=7
            Top n features to be displayed. The importances of the rest are aggregated and displayed under the tag "Rest".
        output_path : str, default=None
            Path to save figure as image.
        savefig_kws : dict, default=None
            Save figure options.
        """
        plot_shap_importances_beeswarm(self.model_, self.X_train_, class_id, n_top=n_top, output_path=output_path,
                                       savefig_kws=savefig_kws)

    def plot_confusion_matrix(self, test=True, sum_stats=True, output_path=None, savefig_kws=None):
        """
        This function makes a pretty plot of an sklearn Confusion Matrix cf using a Seaborn heatmap visualization.

        Parameters
        ---------
        test : boolean, default=True
            If True, returns the confusion matrix calculated on the test set. If False, returns the confusion matrix on
            the train set.
        sum_stats : boolean, default=True
            If True, show precision and recall per class, and global accuracy, appended to the matrix.
        output_path : str, default=None
            Path to save figure as image.
        savefig_kws : dict, default=None
            Save figure options.
        """
        cm = self.confusion_matrix(test, sum_stats)
        plot_confusion_matrix(cm, sum_stats=sum_stats, figsize=(cm.shape[0]+1, cm.shape[0]),
                              output_path=output_path, savefig_kws=savefig_kws)

    def plot_roc_curves(self, test=True, output_path=None, savefig_kws=None):
        """
       Plots ROC curve for every class.

       Parameters
       ---------
       test : boolean, default=True
            If True, returns the confusion matrix calculated on the test set. If False, returns the confusion matrix on
            the train set.
       output_path : str, default=None
           Path to save figure as image.
       savefig_kws : dict, default=None
           Save figure options.
       """
        X = self.X_test_ if test else self.X_train_
        y = self.y_test_ if test else self.y_train_
        plot_roc_curves(X, y, self.model_, output_path=output_path, savefig_kws=savefig_kws)
