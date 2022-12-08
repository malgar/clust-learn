"""Dimensionality Reduction with PCA as reference"""
# Author: Miguel Alvarez-Garcia

import prince

from sklearn.decomposition import PCA, SparsePCA
from sklearn.preprocessing import StandardScaler
from .table_utils import *
from .viz_utils import *


class DimensionalityReduction:
    """
    Dimensionality Reduction class

    Parameters
    ----------
    df : `pandas.DataFrame`
        DataFrame containing the data.
    num_vars : string, list, series, or vector array
        Numerical variable name(s).
    cat_vars : string, list, series, or vector array
        Categorical variable name(s).
    num_algorithm : string, default='pca'
        Technique to be used for dimensionality reduction for numerical variables.
        By default, PCA (Principal Component Analysis) is used.
    cat_algorithm : string, default='mca'
        Technique to be used for dimensionality reduction for categorical variables.
        By default, MCA (Multiple Correspondence Analysis) is used.
    num_kwargs : dictionary, default=None
        Additional keyword arguments to pass to the model used for numerical variables.
    cat_kwargs : dictionary, default=None
        Additional keyword arguments to pass to the model used for categorical variables.
    """

    def __init__(
            self,
            df,
            num_vars=None,
            cat_vars=None,
            num_algorithm='pca',
            cat_algorithm='mca',
            num_kwargs=None,
            cat_kwargs=None
    ):
        self.num_algorithm = str.lower(num_algorithm)
        self.cat_algorithm = str.lower(cat_algorithm)
        self.n_components_ = None
        self.min_explained_variance_ratio_ = 0.5
        self.df = df
        self.num_vars = num_vars
        self.cat_vars = cat_vars
        self.num_trans_ = None
        self.cat_trans_ = None
        self.num_components_ = None
        self.cat_components_ = None
        self.num_kwargs = {} if num_kwargs is None else num_kwargs
        self.cat_kwargs = {} if cat_kwargs is None else cat_kwargs

        # Regardless of the algorithm selected for numerical variables, PCA is used as reference algorithm
        self.pca_ = PCA()

        self.num_model = None
        if self.num_algorithm == 'pca':
            self.num_model = PCA(n_components=None, random_state=42, **self.num_kwargs)
        elif self.num_algorithm in ['spca', 'sparsepca']:
            self.num_model = SparsePCA(n_components=None, random_state=42, **self.num_kwargs)
        else:
            raise RuntimeError('''An error occurred while initializing the algorithm.
                               Check the algorithm name for numerical variables.''')

        self.cat_model = None
        if self.cat_algorithm == 'mca':
            self.cat_model = prince.MCA(n_components=None, n_iter=10, random_state=42, **self.cat_kwargs)
        else:
            raise RuntimeError('''An error occurred while initializing the algorithm.
                               Check the algorithm name for categorical variables.''')

    def transform(self, n_components=None, min_explained_variance_ratio=0.5):
        """
        Transforms a DataFrame df to a lower dimensional space

        Parameters
        ----------
        n_components : int, default=None
            Number components to compute. If None, then `n_components` is set to the number of features.
            Note this number is approximate because numerical and categorical vars are treated independently.
        min_explained_variance_ratio : float, default=0.5
            Minimum explained variance ratio to be achieved. If `n_components` is not None,
            `min_explained_variance_ratio` will be ignored.
            If None: optimal

        Returns
        ----------
        trans : `pandas.DataFrame`
            DataFrame with the transformed data.
        """

        self.n_components_ = n_components
        self.min_explained_variance_ratio_ = min_explained_variance_ratio

        # If the number of components is specified and there are numerical and categorical variables
        # the number of components for each type of variables is distributed proportionally
        n_components_num = None
        n_components_cat = None
        if self.n_components_ is not None:
            if self.num_vars is None:
                n_components_cat = np.minimum(self.n_components_, self.df[self.cat_vars].nunique().sum())
            if self.cat_vars is None:
                n_components_num = np.minimum(self.n_components_, len(self.num_vars))
            if self.num_vars is not None and self.cat_vars is not None:
                n_components_num = int(
                    np.ceil(self.n_components_ * len(self.num_vars) / (len(self.num_vars) + len(self.cat_vars))))
                n_components_cat = int(
                    np.ceil(self.n_components_ * len(self.cat_vars) / (len(self.num_vars) + len(self.cat_vars))))

        trans = pd.DataFrame()
        if self.num_vars is not None:
            self.num_trans_ = self._transform_num(self.df[self.num_vars], n_components_num)
            trans = pd.concat([trans, self.num_trans_], axis=1)
        if self.cat_vars is not None:
            self.cat_trans_ = self._transform_cat(self.df[self.cat_vars], n_components_cat)
            trans = pd.concat([trans, self.cat_trans_], axis=1)

        idx_positions = np.maximum(len(str(trans.shape[1])), 2)
        trans.columns = [f'dim_{str(i+1).zfill(idx_positions)}' for i in range(trans.shape[1])]
        self.num_components_ = list(trans.columns)[:self.num_trans_.shape[1]]
        self.num_trans_.columns = self.num_components_
        self.cat_components_ = list(trans.columns)[-self.cat_trans_.shape[1]:]
        self.cat_trans_.columns = self.cat_components_
        self.n_components_ = len(self.num_components_) + len(self.cat_components_)
        return trans

    def _transform_num(self, df, n_components_num=None):
        # For now, we take PCA as reference because SPCA results depend on the number of components
        sc = StandardScaler()
        
        if self.pca_.n_components is None:
            self.pca_.fit(sc.fit_transform(df))

            if n_components_num is None and self.min_explained_variance_ratio_ is None:
                # Optimal number
                kl = KneeLocator(x=range(1, df.shape[1] + 1),
                                 y=self.pca_.explained_variance_ratio_,
                                 curve='convex',
                                 direction='decreasing')
                n_components_num = kl.knee

            elif n_components_num is None:
                # Based on explained variance
                n_components_num = (self.pca_.explained_variance_ratio_.cumsum() <
                                    self.min_explained_variance_ratio_).sum() + 1

            self.num_model.set_params(n_components=n_components_num)
            self.num_model.fit(sc.fit_transform(df))

        idx_positions = np.maximum(2, len(str(n_components_num)))
        trans = pd.DataFrame(self.num_model.transform(sc.fit_transform(df)),
                             columns=[f'dim_{str(i+1).zfill(idx_positions)}' for i in range(n_components_num)])
        
        # sort by explained variance
        trans = trans[pd.DataFrame(data={'pc': trans.columns, 'explained_var': trans.var().values})
                        .sort_values('explained_var', ascending=False)['pc']]
        trans.columns = [f'dim_{str(i+1).zfill(idx_positions)}' for i in range(trans.shape[1])]
        return trans

    def _transform_cat(self, df, n_components_cat=None):
        self.cat_model.set_params(n_components=df.nunique().sum())
        self.cat_model.fit(df.astype(str))

        explained_variance_ratio = self.cat_model.explained_inertia_ / self.cat_model.explained_inertia_.sum()
        if n_components_cat is None and self.min_explained_variance_ratio_ is None:
            # Optimal number
            kl = KneeLocator(x=range(1, df.nunique().sum() + 1),
                             y=explained_variance_ratio,
                             curve='convex',
                             direction='decreasing')
            n_components_cat = np.maximum(kl.knee-1, 1)

        elif n_components_cat is None:
            # Based on explained variance
            n_components_cat = (explained_variance_ratio.cumsum() <
                                self.min_explained_variance_ratio_).sum() + 1

        trans = self.cat_model.transform(df.astype(str))
        trans = trans[trans.columns[:n_components_cat]]
        idx_positions = np.maximum(2, len(str(n_components_cat)))
        trans.columns = [f'dim_{str(i+1).zfill(idx_positions)}' for i in range(n_components_cat)]
        return trans

    def num_main_contributors(self, thres=0.5, n_contributors=None, dim_idx=None, component_description=None,
                              col_description=None, output_path=None):
        """
        Computes the original numerical variables with the strongest relation to the derived variable(s)
        (measured as Pearson correlation coefficient)

        Parameters
        ----------
        thres : float, default=0.5
            Correlation coefficient threshold to consider one original variable to be a main contributor of a derived
            variable.
        n_contributors : float, default=None
            If n_contributors is passed, the n_contributors original variables with the highest correlation coefficient
            are selected for every derived variable.
            If n_contributors is passed, the correlation coefficient threshold (thres) is ignored.
        dim_idx : int, default=None
            In case only main contributors for derived variable in column position dim_idx are retrieved (starts at 0).
        component_description : str or list
            Description of derived variables. It might be of interest to show a description of the new variables
            on a table for explainability purposes.
        col_description : `pandas.DataFrame`
            DataFrame with two columns: First one with original variable names, and a second one with the description.
            This is also used for explainability purposes.
        output_path : str, default=None
            If an output_path is passed, the resulting DataFame is saved as a CSV file.

        Returns
        ----------
        mc: `pandas.DataFrame`
            DataFrame with the main contributors of every derived variable.
        """
        return num_main_contributors(self.df[self.num_vars], self.num_trans_, thres, n_contributors, dim_idx,
                                     component_description, col_description, output_path)

    def cat_main_contributors(self, thres=0.14, n_contributors=None, dim_idx=None, component_description=None,
                              col_description=None, output_path=None):
        """
        Computes the original categorical variables with the strongest relation to the derived variable(s)
        (measured as correlation ratio)

        Parameters
        ----------
        thres : float, default=0.14
            Correlation ratio threshold to consider one original variable to be a main contributor of a derived
            variable.
        n_contributors : float, default=None
            If n_contributors is passed, the n_contributors original variables with the highest correlation ratio
            are selected for every derived variable.
            If n_contributors is passed, the correlation ratio threshold (thres) is ignored.
        dim_idx : int, default=None
            In case only main contributors for derived variable in column position dim_idx are retrieved (starts at 0).
        component_description : str or list
            Description of derived variables. It might be of interest to show a description of the new variables
            on a table for explainability purposes.
        col_description : `pandas.DataFrame`
            DataFrame with two columns: First one with original variable names, and a second one with the description.
            This is also used for explainability purposes.
        output_path : str, default=None
            If an output_path is passed, the resulting DataFame is saved as a CSV file.

        Returns
        ----------
        mc: `pandas.DataFrame`
            DataFrame with the main contributors of every derived variable.
        """
        return cat_main_contributors(self.df[self.cat_vars], self.cat_trans_, thres, n_contributors, dim_idx,
                                     component_description, col_description, output_path)

    def cat_main_contributors_stats(self, thres=0.14, n_contributors=None, dim_idx=None, output_path=None):
        """
        Computes for every categorical variable's value, the mean and std of the derived variables that are strongly
        related to the categorical variable (based on the correlation ratio)

        Parameters
        ----------
        thres : float, default=0.14
            Correlation ratio threshold to consider one original variable to be a main contributor of a derived
            variable.
        n_contributors : float, default=None
            If n_contributors is passed, the n_contributors original variables with the highest correlation ratio
            are selected for every derived variable.
            If n_contributors is passed, the correlation ratio threshold (thres) is ignored.
        dim_idx : int, default=None
            In case only main contributors for derived variable in column position dim_idx are retrieved (starts at 0).
        output_path : str, default=None
            If an output_path is passed, the resulting DataFame is saved as a CSV file.

        Returns
        ----------
        stats: `pandas.DataFrame`
            DataFrame with the statistics.
        """
        return cat_main_contributors_stats(self.df[self.cat_vars], self.cat_trans_, thres, n_contributors, dim_idx,
                                           output_path)

    def plot_num_explained_variance(self, thres=0.5, plots='all', output_path=None, savefig_kws=None):
        """
        Plot the explained variance (ratio, cumulative, and/or normalized) for numerical variables

        Parameters
        ----------
        thres : float, default=0.5
            Minimum explained cumulative variance ratio.
        plots : str or list, default='all'
            The following plots are supported: ['cumulative', 'ratio', 'normalized']
        output_path : str, default=None
            Path to save figure as image.
        savefig_kws : dict, default=None
            Save figure options.
        """
        plot_explained_variance(self.pca_.explained_variance_ratio_, thres, plots, output_path, savefig_kws)

    def plot_cat_explained_variance(self, thres=0.5, plots='all', output_path=None, savefig_kws=None):
        """
        Plot the explained variance (ratio, cumulative, and/or normalized) for categorical variables

        Parameters
        ----------
        thres : float, default=0.5
            Minimum explained cumulative variance ratio.
        plots : str or list, default='all'
            The following plots are supported: ['cumulative', 'ratio', 'normalized']
        output_path : str, default=None
            Path to save figure as image.
        savefig_kws : dict, default=None
            Save figure options.
        """
        explained_variance_ratio = self.cat_model.explained_inertia_ / self.cat_model.explained_inertia_.sum()
        plot_explained_variance(explained_variance_ratio, thres, plots, output_path, savefig_kws)

    def plot_num_main_contributors(self, thres=0.5, n_contributors=5, dim_idx=None, output_path=None, savefig_kws=None):
        """
        Plot main contributors (original variables with the strongest relation with derived variables) for
        every derived variable

        Parameters
        ----------
        thres : float, default=0.5
            Minimum Pearson correlation coefficient to consider an original and a derived variable to be strongly
            related.
        n_contributors : int, default=5
            Number of contributors by derived variables (the ones with the strongest correlation coefficient
            are shown).
        dim_idx : int, default=None
            In case only main contributors for derived variable in column position dim_idx are retrieved (starts at 0).
        output_path : str, default=None
            Path to save figure as image.
        savefig_kws : dict, default=None
            Save figure options.
        """
        plot_num_main_contributors(self.df[self.num_vars], self.num_trans_, thres, n_contributors, dim_idx, output_path,
                                   savefig_kws)

    def plot_cat_main_contributor_distribution(self, thres=0.14, n_contributors=None, dim_idx=None, output_path=None,
                                               savefig_kws=None):
        """
        Plot main contributors (original variables with the strongest relation with derived variables) for
        every derived variable

        Parameters
        ----------
        thres : float, default=0.5
             Minimum correlation ratio to consider an original and a derived variable to be strongly related.
        n_contributors : int, default=5
            Number of contributors by derived variables (the ones with the strongest correlation coefficient
            are shown).
        dim_idx : int, default=None
            In case only main contributors for derived variable in column position dim_idx are retrieved (starts at 0).
        output_path : str, default=None
            Path to save figure as image.
        savefig_kws : dict, default=None
            Save figure options.
        """
        plot_cat_main_contributor_distribution(self.df[self.cat_vars], self.cat_trans_, thres, n_contributors, dim_idx,
                                               output_path, savefig_kws)

    def plot_cumulative_explained_var_comparison(self, thres=None, output_path=None, savefig_kws=None):
        """
        Plots comparison of cumulative explained variance between two techniques.

        Parameters
        ----------
        thres : float, default=None
             Reference threshold for cumulative explained variance ratio. (For styling purposes).
        output_path : str, default=None
            Path to save figure as image.
        savefig_kws : dict, default=None
            Save figure options.
        """
        plot_cumulative_explained_var_comparison(self.pca_.explained_variance_ratio_.cumsum(),
                                                 (self.num_trans_.var().values/len(self.num_vars)).cumsum(),
                                                 name1='pca', name2=self.num_algorithm, thres=thres,
                                                 output_path=output_path, savefig_kws=savefig_kws)
