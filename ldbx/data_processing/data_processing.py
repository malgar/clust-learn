"""Data processing"""
# Author: Miguel Alvarez

from utils import *


class DataProcessing(object):

    def __init__(self,
                 df,
                 num_vars=None,
                 cat_vars=None):

        self.df = df
        self.num_vars = num_vars
        self.cat_vars = cat_vars
        self._imputation_pairs = None

    # TODO: Decide whether a class makes sense here. I would say it doesn't as long as we expose the right functions
    #
    # TODO: Deletion of variables with a pct of missing values above a given threshold

    # TODO: This is a high level class. If the user wants to know what happens in every step, call functions
    # independently
    def impute_missing_values(self):
        # One-to-one model based imputation for strongly related variables
        self._imputation_pairs = imputation_pairs(self.df, self.num_vars, self.cat_vars)
        df_imp = impute_missing_values_with_highly_related_pairs(self.df, self._imputation_pairs, self.num_vars,
                                                                 self.cat_vars)

        mi_scores = mutual_information_pair_scores(df_imp, self.num_vars, self.cat_vars)
        nopdes, edges, comp = variable_graph_partitioning(mi_scores, thres=0.05)

        df_imp2 = hot_deck_imputation(df_imp, self.num_vars + self.cat_vars, k=8,
                                      partitions=list(np.array(comp)[np.array(list(map(len, comp))) >= 4]))
        df_imp2_clean = df_imp2[df_imp2.isnull().sum(1) < df_imp2.shape[1] / 3]
        df_imp3 = hot_deck_imputation(df_imp2_clean, self.num_vars + self.cat_vars, k=8)

    def compute_missing(self, normalize=True):
        """
        Calculates the pct/count of missing values per column.

        Parameters
        ----------
        normalize : boolean, default=True

        Returns
        ----------
        missing_df : `pandas.DataFrame`
            DataFrame with the pct/counts of missing values per column.
        """
        return compute_missing(self.df, normalize=normalize)

    # TODO: Decide what to pass as arguments
    def plot_imputation_distribution_assessment(self, df_prior, df_posterior, imputed_vars, sample_frac=1.0, prior_kws=None,
                                                posterior_kws=None, output_path=None, savefig_kws=None):
        """
        Plots a distribution comparison of each variable with imputed variables, before and after imputation.

        Parameters
        ----------
        df_prior : `pandas.DataFrame`
            DataFrame containing the data before imputation.
        df_posterior : `pandas.DataFrame`
            DataFrame containing the data after imputation.
        imputed_vars : list
            List of variables with imputed variables.
        sample_frac : float, default=1.0
            If < 1 a random sample of every pair of variables will be plotted.
        {prior,posterior}_kws : dict, default=None
            Additional keyword arguments to pass to the kdeplot (https://seaborn.pydata.org/generated/seaborn.kdeplot.html).
        output_path : str, default=None
            Path to save figure as image.
        savefig_kws : dict, default=None
            Save figure options.
        """
        prior_kws = prior_kws if prior_kws else dict(color='#7F3C8D')
        posterior_kws = posterior_kws if posterior_kws else dict(color='#11A579')

        ncols = min(4, len(imputed_vars))
        nrows = int(np.ceil(len(imputed_vars) / ncols))

        fig, axs = plt.subplots(nrows, ncols, figsize=(4 * ncols, 4 * nrows))
        df_prior_sample = df_prior.sample(frac=sample_frac, random_state=42)
        i = 0
        for ivar in imputed_vars:
            ax = axs[i // ncols, i % ncols]
            sns.kdeplot(ivar, data=df_prior_sample, label='Before imputation', ax=ax, **prior_kws)
            sns.kdeplot(ivar, data=df_posterior.loc[df_prior_sample.index], label='After imputation', ax=ax,
                        **posterior_kws)
            ax.legend()
            i += 1

        fig.tight_layout(pad=2)
        savefig(output_path=output_path, savefig_kws=savefig_kws)

    def remove_outliers(self, iforest_kws=None):
        """
        Removes outliers using the Isolation Forest algorithm
        (https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.IsolationForest.html).

        Parameters
        ----------
        iforest_kws : dict, default=None
            IsolationForest algorithm hyperparameters.

        Returns
        ----------
        df_inliers : `pandas.DataFrame`
            DataFrame with inliers (i.e. observations that are not outliers).
        df_outliers : `pandas.DataFrame`
            DataFrame with outliers.
        """
        if iforest_kws is None:
            iforest_kws = dict(max_samples=0.8, max_features=0.8, bootstrap=False)
        outlier_if = IsolationForest(**iforest_kws)
        outlier_flag = outlier_if.fit_predict(self.df[self.num_vars + self.cat_vars])
        return self.df[outlier_flag > 0], self.df[outlier_flag < 0]
