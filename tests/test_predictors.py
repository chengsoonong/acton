#!/usr/bin/env python3

"""
test_predictors
----------------------------------

Tests for `predictors` module.
"""

import os.path
import sys
import tempfile
import unittest
import unittest.mock

import acton.database
import acton.predictors
import acton.proto.wrappers
from acton.proto.predictors_pb2 import Labels
import numpy
import sklearn.linear_model


class TestPredictorInput(unittest.TestCase):

    def setUp(self):
        # Make a protobuf.
        self.labels = Labels()
        self.instance = self.labels.instance.add()
        self.instance.id = 'instance id'
        self.instance.label.extend([0, 1])
        self.labels.n_labellers = 2
        self.labels.n_label_dimensions = 1
        self.labels.dtype = 'float32'
        self.labels.db_class = 'HDF5Database'
        self.tempdir = tempfile.TemporaryDirectory()
        self.path = os.path.join(self.tempdir.name, 'predin.proto')
        self.labels.db_path = os.path.join(self.tempdir.name, 'test.h5')
        with open(self.path, 'wb') as f:
            f.write(self.labels.SerializeToString())
        self.features = numpy.array([2, 5]).reshape((1, 2))

    def tearDown(self):
        self.tempdir.cleanup()

    def test_labels(self):
        """PredictorInput gives a label array."""
        predictor_input = acton.proto.wrappers.PredictorInput(self.path)
        self.assertTrue(numpy.all(
            predictor_input.labels == numpy.array([[[0]], [[1]]])))  # T x N x F
        self.assertTrue(numpy.all(
            predictor_input.labels == numpy.array([[[0]], [[1]]])))  # T x N x F
        # Second assertion because the property is cached after the first run.

    @unittest.mock.patch.dict(acton.database.DATABASES, values={
        'HDF5Database': unittest.mock.MagicMock()
    })
    def test_features(self):
        """PredictorInput gives a features array."""
        # Setup mock.
        for DB in acton.database.DATABASES:
            DB = acton.database.DATABASES[DB]
            db = DB().__enter__.return_value
            db.read_features.return_value = self.features

        predictor_input = acton.proto.wrappers.PredictorInput(self.path)
        self.assertTrue(numpy.allclose(
            self.features,
            predictor_input.features))


class TestIntegrationCommittee(unittest.TestCase):
    """Integration test for Committee."""

    def setUp(self):
        # Make a protobuf.
        self.labels = Labels()

        self.instance_1 = self.labels.instance.add()
        self.instance_1.id = b'instance id 1'
        self.instance_1.label.append(float(0))

        self.instance_2 = self.labels.instance.add()
        self.instance_2.id = b'instance id 2'
        self.instance_2.label.append(float(1))

        self.labels.n_labellers = 1
        self.labels.n_label_dimensions = 1
        self.labels.dtype = 'float64'
        self.labels.db_class = 'HDF5Database'

        self.tempdir = tempfile.TemporaryDirectory()
        self.path = os.path.join(self.tempdir.name, 'predin.proto')
        self.labels.db_path = os.path.join(self.tempdir.name, 'test.h5')
        with open(self.path, 'wb') as f:
            f.write(self.labels.SerializeToString())

        self.n_instances = 2
        self.features = numpy.array([2, 5, 3, 7]).reshape((self.n_instances, 2))
        with acton.database.HDF5Database(
                self.labels.db_path, label_dtype=self.labels.dtype,
                feature_dtype='int32') as db:
            db.write_features([self.instance_1.id, self.instance_2.id],
                              self.features)

    def tearDown(self):
        self.tempdir.cleanup()

    def testAll(self):
        """Committee can be used with PredictorInput."""
        pred_input = acton.proto.wrappers.PredictorInput(self.labels)
        lrc = acton.predictors.Committee(
            acton.predictors.from_class(
                sklearn.linear_model.LogisticRegression),
            n_classifiers=10)
        lrc.fit(pred_input.features, pred_input.labels.reshape((-1, 1)))
        probs = lrc.predict(pred_input.features)
        self.assertEqual((self.n_instances, 10), probs.shape)


class TestSklearnWrapper(unittest.TestCase):
    """Integration test for scikit-learn wrapper functions."""

    def setUp(self):
        # Make a protobuf.
        self.labels = Labels()

        self.instance_1 = self.labels.instance.add()
        self.instance_1.id = b'instance id 1'
        self.instance_1.label.append(float(0))

        self.instance_2 = self.labels.instance.add()
        self.instance_2.id = b'instance id 2'
        self.instance_2.label.append(float(1))

        self.labels.n_labellers = 1
        self.labels.n_label_dimensions = 1
        self.labels.dtype = 'float64'
        self.labels.db_class = 'HDF5Database'

        self.tempdir = tempfile.TemporaryDirectory()
        self.path = os.path.join(self.tempdir.name, 'predin.proto')
        self.labels.db_path = os.path.join(self.tempdir.name, 'test.h5')
        with open(self.path, 'wb') as f:
            f.write(self.labels.SerializeToString())

        self.n_instances = 2
        self.features = numpy.array([2, 5, 3, 7]).reshape((self.n_instances, 2))
        with acton.database.HDF5Database(
                self.labels.db_path, label_dtype=self.labels.dtype,
                feature_dtype='int32') as db:
            db.write_features([self.instance_1.id, self.instance_2.id],
                              self.features)

    def tearDown(self):
        self.tempdir.cleanup()

    def testFromInstance(self):
        """from_instance wraps a scikit-learn classifier."""
        # The main point of this test is to check nothing crashes.
        classifier = sklearn.linear_model.LogisticRegression()
        pred_input = acton.proto.wrappers.PredictorInput(self.labels)
        predictor = acton.predictors.from_instance(classifier)
        predictor.fit(pred_input.features, pred_input.labels.reshape((-1, 1)))
        probs = predictor.predict(pred_input.features)
        self.assertEqual((2, 1), probs.shape)

    def testFromClass(self):
        """from_class wraps a scikit-learn classifier."""
        # The main point of this test is to check nothing crashes.
        Classifier = sklearn.linear_model.LogisticRegression
        pred_input = acton.proto.wrappers.PredictorInput(self.labels)
        Predictor = acton.predictors.from_class(Classifier)
        predictor = Predictor(C=50.0)
        predictor.fit(pred_input.features, pred_input.labels.reshape((-1, 1)))
        probs = predictor.predict(pred_input.features)
        self.assertEqual((2, 1), probs.shape)
