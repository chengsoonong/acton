"""Tests for kde_predictor."""

import unittest

import acton.kde_predictor
import numpy
import sklearn.utils.estimator_checks


class TestKDEClassifier(unittest.TestCase):

    def test_sklearn_interface(self):
        """KDEClassifier implements the scikit-learn interface."""
        sklearn.utils.estimator_checks.check_estimator(
            acton.kde_predictor.KDEClassifier)

    def test_softmax(self):
        """_softmax correctly evaluates a softmax on array input."""
        for axis in range(2):
            for _ in range(100):
                array = numpy.random.random(size=(100, 100)) * 1000 - 500
                softmax = acton.kde_predictor.KDEClassifier._softmax(
                    array, axis=axis)
                for i in softmax.sum(axis=axis):
                    self.assertAlmostEqual(i, 1)
