"""Tests for regression functionality."""

import os.path
import tempfile
import unittest

import acton.database
import acton.predictors
import numpy


class TestRegression(unittest.TestCase):
    """Acton supports regression."""

    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.db_path = os.path.join(self.tempdir.name, 'db.h5')

    def tearDown(self):
        self.tempdir.cleanup()

    def test_linear_regression(self):
        """LinearRegression predictor can find a linear fit."""
        # Some sample data.
        numpy.random.seed(0)
        xs = numpy.linspace(0, 1, 100)
        ys = 2 * xs - 1
        noise = numpy.random.normal(size=xs.shape, scale=0.2)
        xs = xs.reshape((-1, 1))
        ts = (ys + noise).reshape((1, -1, 1))
        ids = list(range(100))

        with acton.database.ManagedHDF5Database(self.db_path) as db:
            db.write_features(ids, xs)
            db.write_labels([0], ids, ts)
            lr = acton.predictors.PREDICTORS['LinearRegression'](db)
            lr.fit(ids)
            predictions = lr.predict(ids)
            print(ys, predictions.ravel())
            self.assertTrue(numpy.allclose(ys, predictions.ravel(), atol=0.2))
