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

import acton.database
import acton.proto.wrappers
import numpy


class TestLabelPool(unittest.TestCase):
    """Tests the LabelPool wrapper."""

    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.path = os.path.join(self.tempdir.name, 'labelpool.proto')
        self.db_path = os.path.join(self.tempdir.name, 'db.txt')
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
        with acton.database.ASCIIReader(self.db_path, **self.db_kwargs) as db:
            lp = acton.proto.wrappers.LabelPool.make(ids=ids, db=db)
        self.assertTrue(([b'0', b'1'] == lp.labels.ravel()).all())
        self.assertEqual([0, 2], lp.ids)
        with lp.DB() as db:
            self.assertEqual([0, 1, 2], db.get_known_instance_ids())
        self.assertEqual('ASCIIReader', lp.proto.db.class_name)


class TestPredictions(unittest.TestCase):
    """Tests the Predictions wrapper."""

    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.path = os.path.join(self.tempdir.name, 'predictions.proto')
        self.db_path = os.path.join(self.tempdir.name, 'db.txt')
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
        predicted_ids = [0, 2]
        labelled_ids = [1, 2]
        predictions = numpy.array([0.1, 0.5, 0.5, 0.9]).reshape((2, 2, 1))
        with acton.database.ASCIIReader(self.db_path, **self.db_kwargs) as db:
            preds = acton.proto.wrappers.Predictions.make(
                predicted_ids=predicted_ids, labelled_ids=labelled_ids,
                predictions=predictions, db=db)
        self.assertEqual([0, 2], preds.predicted_ids)
        self.assertEqual([1, 2], preds.labelled_ids)
        with preds.DB() as db:
            self.assertEqual([0, 1, 2], db.get_known_instance_ids())
        self.assertTrue(numpy.allclose(predictions, preds.predictions))


class TestRecommendations(unittest.TestCase):
    """Tests the Recommendations wrapper."""

    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.path = os.path.join(self.tempdir.name, 'recommendations.proto')
        self.recommender = 'UncertaintyRecommender'
        self.db_path = os.path.join(self.tempdir.name, 'db.txt')
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
        recommended_ids = [0, 2]
        labelled_ids = [1, 2]
        with acton.database.ASCIIReader(self.db_path, **self.db_kwargs) as db:
            recs = acton.proto.wrappers.Recommendations.make(
                recommended_ids=recommended_ids, labelled_ids=labelled_ids,
                recommender=self.recommender, db=db)
        self.assertEqual([0, 2], recs.recommendations)
        self.assertEqual([1, 2], recs.labelled_ids)
        with recs.DB() as db:
            self.assertEqual([0, 1, 2], db.get_known_instance_ids())
