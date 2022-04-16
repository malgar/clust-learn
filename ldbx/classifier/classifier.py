"""Classification class"""

import logging
import numpy as np
import utils

from sklearn import base
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

from viz_utils import *


class Classifier(base.BaseEstimator, base.TransformerMixin):

    def __init__(self, df, predictors, target):
        self.df = df
        self.original_features = predictors
        self.filtered_features = None
        self.target = target
        self.model_ = None
        self.X_train_, self.X_test_, self.y_train_, self.y_test_ = None
        self.grid_result_ = None

        # Initialize logger
        logging.basicConfig(format='%(asctime)s %(levelname)-8s %(name)s: %(message)s',
                            level=logging.INFO,
                            datefmt='%Y-%m-%d %H:%M:%S')

        self.logger = logging.getLogger(__name__)

    def train_model(self, model=None, feature_selection=True, features_to_keep=[],
                    feature_selection_model=None, hyperparameter_tuning=False, param_grid=None,
                    train_size=0.8):
        # Feature selection
        if feature_selection:
            self.logger.info('Running feature selection...')
            if feature_selection_model is None:
                min_samples_leaf = int(np.ceil(self.df.shape[0] * 0.05))
                feature_selection_model = RandomForestClassifier(max_depth=10,  min_samples_leaf=min_samples_leaf)

            self.filtered_features = utils.feature_selection(self.df, self.original_features, self.target,
                                                             features_to_keep, feature_selection_model)
        else:
            self.filtered_features = self.original_features.copy()

        # Data is split into train and test sets
        X = self.df[self.filtered_features]
        y = self.target
        self.X_train_, self.X_test_, self.y_train_, self.y_test_ = train_test_split(X, y, train_size=train_size)

        # Model instantiation
        if model is None:
            model = XGBClassifier(eval_metric='auc')
        self.model_ = model

        # Hyperparameter tuning
        if hyperparameter_tuning:
            self.logger.info('Running hyperparameter tuning...')
            self.grid_result_ = utils.hyperparameter_tuning(self.X_train_, self.y_train_, self.model_, param_grid)
            self.model_.set_params(**self.grid_result_.best_params_)

        # Model training
        self.logger.info('Training model...')
        self.model_.fit(self.X_train_, self.y_train_)
        self.logger.info('DONE!')

    @property
    def feature_importances(self):
        return utils.shap_importances(self.model_, self.X_train_)

    @property
    def hyperparameter_tuning_metrics(self, output_path=None):
        htm = pd.DataFrame(self.grid_result_.cv_results_['params'])
        htm.columns = pd.MultiIndex.from_product([[f'{self.model_.__class__.__name__} Hyperparameters'], htm.columns])
        htm[('Performance metrics', 'mean_test_score')] = self.grid_result_.cv_results_['mean_test_score']
        htm[('Performance metrics', 'std_test_score')] = self.grid_result_.cv_results_['std_test_score']

        if output_path is not None:
            htm.to_csv(output_path, index=False)

        return htm

    def confusion_matrix(self, test=True, sum_stats=True, output_path=None):
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
        X = self.X_test_ if test else self.X_train_
        y = self.y_test_ if test else self.y_train_

        report = pd.DataFrame(classification_report(y, self.model_.predict(X), output_dict=True)).transpose()

        if output_path is not None:
            # In this case we do want to keep the index
            report.to_csv(output_path)

        return report

    def plot_shap_importances(self, n_top=7, output_path=None, savefig_kws=None):
        plot_shap_importances(self.model_, self.X_train_, n_top=n_top, output_path=output_path, savefig_kws=savefig_kws)

    def plot_confusion_matrix(self, test=True, sum_stats=True, output_path=None, savefig_kws=None):
        cm = self.confusion_matrix(test, sum_stats)
        plot_confusion_matrix(cm, sum_stats=sum_stats, figsize=(cm.shape[0]+1, cm.shape[0]), output_path=output_path,
                              savefig_kws=savefig_kws)

    def plot_roc_curves(self, test=True, output_path=None, savefig_kws=None):
        X = self.X_test_ if test else self.X_train_
        y = self.y_test_ if test else self.y_train_
        plot_roc_curves(X, y, self.model_, output_path=output_path, savefig_kws=savefig_kws)
