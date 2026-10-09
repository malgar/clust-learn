from unittest import TestCase

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

from clearn.classifier.viz_utils import _confusion_matrix_box_labels
from clearn.clustering import Clustering
from clearn.dimensionality_reduction import DimensionalityReduction

# Non-contiguous index, as returned by `remove_outliers()`
GAPPED_INDEX = pd.Index([i for i in range(123) if i not in (3, 7, 11)])


class TestPreprocessing(TestCase):
    def test_imputation(self):
        self.assertEqual(0, 0)

    def test_outliers(self):
        self.assertEqual(0, 0)


class TestDimReduction(TestCase):
    def test_dim_reduction(self):
        self.assertEqual(0, 0)

    def test_transform_non_contiguous_index(self):
        rng = np.random.default_rng(0)
        n = len(GAPPED_INDEX)
        df = pd.DataFrame({'x1': rng.normal(size=n), 'x2': rng.normal(size=n), 'x3': rng.normal(size=n),
                           'c1': rng.choice(list('abc'), n), 'c2': rng.choice(list('de'), n)}, index=GAPPED_INDEX)

        trans = DimensionalityReduction(df, num_vars=['x1', 'x2', 'x3'], cat_vars=['c1', 'c2']).transform()
        trans_reset = DimensionalityReduction(df.reset_index(drop=True), num_vars=['x1', 'x2', 'x3'],
                                              cat_vars=['c1', 'c2']).transform()

        self.assertTrue(trans.index.equals(df.index))
        self.assertFalse(trans.isna().any().any())
        np.testing.assert_allclose(trans.values, trans_reset.values)


class TestClustering(TestCase):
    def test_clustering(self):
        self.assertEqual(0, 0)

    def test_non_contiguous_index(self):
        rng = np.random.default_rng(0)
        n = len(GAPPED_INDEX)
        X = rng.normal(size=(n, 2))
        cat = rng.choice(list('ab'), n)
        weights = rng.uniform(0.5, 2, n)

        results = []
        for index in [pd.RangeIndex(n), GAPPED_INDEX]:
            cl = Clustering(pd.DataFrame(X, columns=['dim_01', 'dim_02'], index=index),
                            algorithms=[KMeans(random_state=42, n_init=10)])
            cl.compute_clusters(n_clusters=3, weights=weights)
            self.assertTrue(cl.df.index.equals(index))
            cat_series = pd.Series(cat, index=index)
            results.append([cl.describe_clusters(statistics=['mean', 'wmean']),
                            cl.describe_clusters_cat(cat_series, 'cat'),
                            cl.describe_clusters_cat(cat_series, 'cat', use_weights=True)])

        # Results must not depend on the index labels
        for res_range, res_gapped in zip(*results):
            np.testing.assert_allclose(res_range.values.astype(float), res_gapped.values.astype(float))
        self.assertEqual(results[1][1].values.sum(), n)


class TestClassifier(TestCase):
    def test_classifier(self):
        self.assertEqual(0, 0)

    @staticmethod
    def _cf_with_sum_stats(counts):
        # Same construction as `Classifier.confusion_matrix(sum_stats=True)`
        cm = pd.DataFrame(counts)
        recall = np.diag(cm) / cm.sum(1)
        precision = np.diag(cm) / cm.sum()
        accuracy = np.diag(cm).sum() / cm.sum().sum()
        cm['recall'] = recall
        cm = cm.transpose()
        cm['precision'] = list(precision) + [accuracy]
        return cm.transpose()

    def test_confusion_matrix_percentages(self):
        counts = np.array([[143, 10, 11],
                           [12, 103, 13],
                           [7, 11, 93]])
        cf = self._cf_with_sum_stats(counts)
        box_labels = _confusion_matrix_box_labels(cf, sum_stats=True)

        n = counts.shape[0]
        percentages = np.array([[float(box_labels[i, j].split('\n')[-1].rstrip('%')) for j in range(n)]
                                for i in range(n)])

        # Cell percentages are relative to the total count, not to counts plus appended recall/precision/accuracy
        np.testing.assert_allclose(percentages, 100 * counts / counts.sum(), atol=0.005)
        self.assertAlmostEqual(percentages.sum(), 100, delta=0.005 * counts.size)
