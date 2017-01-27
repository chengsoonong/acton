#!/usr/bin/env python3

"""
test_proto_wrappers
----------------------------------

Tests for `proto.wrappers` module.
"""

import os.path
import tempfile
import unittest
import unittest.mock

import acton.proto.wrappers
import numpy


class TestLabelPool(unittest.TestCase):
    """Tests the LabelPool wrapper."""

    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.path = os.path.join(self.tempdir.name, 'labelpool.proto')
        self.db_path = os.path.join(self.tempdir.name, 'db.txt')
        self.db_class = 'ASCIIReader'
        self.db_kwargs = {
            'feature_cols': ['feature'],
            'label_col': 'label',
        }
        with open(self.db_path, 'w') as f:
            f.write('id\tfeature\tlabel\n')
            f.write('0\t0.2\t0\n')
            f.write('1\t0.1\t0\n')
            f.write('2\t0.6\t1\n')

    def tearDown(self):
        self.tempdir.cleanup()

    def test_integration(self):
        """LabelPool.make returns a LabelPool with correct values."""
        ids = [0, 2]
        lp = acton.proto.wrappers.LabelPool.make(
            ids=ids, db_path=self.db_path, db_class=self.db_class,
            db_kwargs=self.db_kwargs)
        self.assertTrue(([b'0', b'1'] == lp.labels.ravel()).all())
        self.assertEqual([0, 2], lp.ids)
        with lp.DB() as db:
            self.assertEqual([0, 1, 2], db.get_known_instance_ids())


class TestPredictions(unittest.TestCase):
    """Tests the Predictions wrapper."""

    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.path = os.path.join(self.tempdir.name, 'predictions.proto')
        self.db_path = os.path.join(self.tempdir.name, 'db.txt')
        self.db_class = 'ASCIIReader'
        self.db_kwargs = {
            'feature_cols': ['feature'],
            'label_col': 'label',
        }
        with open(self.db_path, 'w') as f:
            f.write('id\tfeature\tlabel\n')
            f.write('0\t0.2\t0\n')
            f.write('1\t0.1\t0\n')
            f.write('2\t0.6\t1\n')

    def tearDown(self):
        self.tempdir.cleanup()

    def test_integration(self):
        """Predictions.make returns a Predictions with correct values."""
        ids = [0, 2]
        predictions = numpy.array([0.1, 0.5, 0.5, 0.9]).reshape((2, 2, 1))
        preds = acton.proto.wrappers.Predictions.make(
            predicted_ids=ids, labelled_ids=ids, predictions=predictions,
            db_path=self.db_path, db_class=self.db_class,
            db_kwargs=self.db_kwargs)
        self.assertEqual([0, 2], preds.ids)
        with preds.DB() as db:
            self.assertEqual([0, 1, 2], db.get_known_instance_ids())
        self.assertTrue(numpy.allclose(predictions, preds.predictions))


class TestRecommendations(unittest.TestCase):
    """Tests the Recommendations wrapper."""

    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.path = os.path.join(self.tempdir.name, 'recommendations.proto')
        self.db_path = os.path.join(self.tempdir.name, 'db.txt')
        self.db_class = 'ASCIIReader'
        self.db_kwargs = {
            'feature_cols': ['feature'],
            'label_col': 'label',
        }
        with open(self.db_path, 'w') as f:
            f.write('id\tfeature\tlabel\n')
            f.write('0\t0.2\t0\n')
            f.write('1\t0.1\t0\n')
            f.write('2\t0.6\t1\n')

    def tearDown(self):
        self.tempdir.cleanup()

    def test_integration(self):
        """Recommendations.make returns Recommendations with correct values."""
        ids = [0, 2]
        recs = acton.proto.wrappers.Recommendations.make(
            recommended_ids=ids, labelled_ids=ids, db_path=self.db_path,
            db_class=self.db_class, db_kwargs=self.db_kwargs)
        self.assertEqual([0, 2], recs.recommendations)
        with recs.DB() as db:
            self.assertEqual([0, 1, 2], db.get_known_instance_ids())
