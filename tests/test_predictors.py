#!/usr/bin/env python3

"""
test_predictors
----------------------------------

Tests for `predictors` module.
"""

import os.path
import tempfile
import unittest
import unittest.mock

import acton.database
import acton.predictors
import acton.proto.wrappers
from acton.proto.acton_pb2 import LabelPool
import numpy
import sklearn.linear_model


class TestIntegrationCommittee(unittest.TestCase):
    """Integration test for Committee."""

    def setUp(self):
        # Make a protobuf.
        self.ids = LabelPool()

        self.ids.id.append(1)
        self.ids.id.append(3)

        self.ids.db.class_name = 'ManagedHDF5Database'

        self.tempdir = tempfile.TemporaryDirectory()
        self.path = os.path.join(self.tempdir.name, 'predin.proto')
        self.ids.db.path = os.path.join(self.tempdir.name, 'test.h5')
        with open(self.path, 'wb') as f:
            f.write(self.ids.SerializeToString())

        self.n_instances = 2
        self.features = numpy.array([2, 5, 3, 7]).reshape((self.n_instances, 2))
        with acton.database.ManagedHDF5Database(
                self.ids.db.path,
                feature_dtype='int32') as db:
            db.write_features(self.ids.id,
                              self.features)
            labels = numpy.array([0, 1]).reshape((1, -1, 1))
            db.write_labels([0], self.ids.id, labels)

    def tearDown(self):
        self.tempdir.cleanup()

    def testAll(self):
        """Committee can be used with LabelPool."""
        pred_input = acton.proto.wrappers.LabelPool(self.ids)
        with pred_input.DB() as db:
            lrc = acton.predictors.Committee(
                acton.predictors.from_class(
                    sklearn.linear_model.LogisticRegression), db,
                n_classifiers=10)
            ids = pred_input.ids
            lrc.fit(ids)
            probs = lrc.predict(ids)
            self.assertEqual((self.n_instances, 10, 2), probs.shape)


class TestSklearnWrapper(unittest.TestCase):
    """Integration test for scikit-learn wrapper functions."""

    def setUp(self):
        # Make a protobuf.
        self.ids = LabelPool()

        self.ids.id.append(1)
        self.ids.id.append(3)

        self.ids.db.class_name = 'ManagedHDF5Database'

        self.tempdir = tempfile.TemporaryDirectory()
        self.path = os.path.join(self.tempdir.name, 'predin.proto')
        self.ids.db.path = os.path.join(self.tempdir.name, 'test.h5')
        with open(self.path, 'wb') as f:
            f.write(self.ids.SerializeToString())

        self.n_instances = 2
        self.features = numpy.array([2, 5, 3, 7]).reshape((self.n_instances, 2))
        with acton.database.ManagedHDF5Database(
                self.ids.db.path,
                feature_dtype='int32') as db:
            db.write_features(self.ids.id,
                              self.features)
            labels = numpy.array([0, 1]).reshape((1, -1, 1))
            db.write_labels([0], self.ids.id, labels)

    def tearDown(self):
        self.tempdir.cleanup()

    def testFromInstance(self):
        """from_instance wraps a scikit-learn classifier."""
        # The main point of this test is to check nothing crashes.
        classifier = sklearn.linear_model.LogisticRegression()
        pred_input = acton.proto.wrappers.LabelPool(self.ids)

        with pred_input.DB() as db:
            predictor = acton.predictors.from_instance(classifier, db)
            ids = pred_input.ids
            predictor.fit(ids)
            probs = predictor.predict(ids)
            self.assertEqual((2, 1, 2), probs.shape)

    def testFromClass(self):
        """from_class wraps a scikit-learn classifier."""
        # The main point of this test is to check nothing crashes.
        Classifier = sklearn.linear_model.LogisticRegression
        pred_input = acton.proto.wrappers.LabelPool(self.ids)

        with pred_input.DB() as db:
            Predictor = acton.predictors.from_class(Classifier)
            predictor = Predictor(db, C=50.0)
            ids = pred_input.ids
            predictor.fit(ids)
            probs = predictor.predict(ids)
            self.assertEqual((2, 1, 2), probs.shape)
