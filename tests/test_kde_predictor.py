"""Tests for kde_predictor."""

import unittest

import acton.kde_predictor
import sklearn.utils.estimator_checks


class TestSklearnInterface(unittest.TestCase):

    def test_sklearn_interface(self):
        """KDEClassifier implements the scikit-learn interface."""
        sklearn.utils.estimator_checks.check_estimator(
            acton.kde_predictor.KDEClassifier)
