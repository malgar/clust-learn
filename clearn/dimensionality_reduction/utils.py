"""Utils for clustering"""
# Author: Miguel Alvarez-Garcia

import numpy as np


# This is (temporarily) needed because prince MCA does no longer apply Benzecri or Greenacre's corrections
def apply_benzecri_eigenvalue_correction(eigenvalues, K):
    """
    Applies Benzecri's correction.
    Benzécri JP (1979). “Sur le Calcul des Taux d’Inertie dans l’Analyse d’un Questionnaire, Addendum et Erratum à
    [BIN. MULT.].” Cahiers de l’Analyse des Données, 4(3), 377–378.

    Parameters
    ----------
    eigenvalues : `numpy.array`
    K : int
        Number of categorical variables

    Returns
    ----------
    `numpy.array` with corrected eigenvalues
    """
    return np.array([(K/(K-1.)*(lamb - 1./K))**2 if lamb > 1./K else 0 for lamb in eigenvalues])


def compute_greenacre_inertia(eigenvalues, K, J):
    """
    Computes Greenacre inertia.
    Greenacre M (1993). Correspondence Analysis in Practice. Academic Press, London

    Parameters
    ----------
    eigenvalues : `numpy.array`
    K : int
        Number of categorical variables
    J : int
        Number of different categories of all categorical variables combined

    Returns
    ----------
    greenacre_inertia : `numpy.array`
        Greenacre inertia
    """
    greenacre_inertia = K / (K - 1.) * (sum(eigenvalues ** 2) - (J - K) / K ** 2)
    return greenacre_inertia
