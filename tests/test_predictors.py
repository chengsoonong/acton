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
        self.labels.db_class = 'ManagedHDF5Database'
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
        'ManagedHDF5Database': unittest.mock.MagicMock()
    })
    def test_features(self):
        """PredictorInput gives a features array."""
        # Setup mock.
        DB = acton.database.DATABASES['ManagedHDF5Database']
        db = DB().__enter__.return_value
        db.read_features.return_value = self.features

        predictor_input = acton.proto.wrappers.PredictorInput(self.path)
        self.assertTrue(numpy.allclose(
            self.features,
            predictor_input.features))


class TestFromPredictions(unittest.TestCase):
    """Tests from_predictions."""

    def setUp(self):
        self.n_predictors = 10
        self.n_instances = 20
        self.n_prediction_dimensions = 5
        self.ids = [str(i).encode('ascii') for i in range(self.n_instances)]
        self.predictions = numpy.random.random(
            size=(self.n_predictors, self.n_instances,
                  self.n_prediction_dimensions))

    def test_protobuf(self):
        """from_predictions converts predictions into a protobuf."""
        proto = acton.proto.wrappers.from_predictions(
            self.ids, self.predictions,
            predictor='test', db_path='path', db_class='ManagedHDF5Database')
        self.assertEqual(self.n_instances, len(proto.proto.prediction))
        # Check the predictions and ID for each instance.
        for i in range(self.n_instances):
            self.assertTrue(numpy.allclose(
                self.predictions[:, i],
                numpy.array(proto.proto.prediction[i].prediction).reshape((
                    proto.proto.n_predictors,
                    proto.proto.n_prediction_dimensions,
                ))
            ))
            self.assertEqual(self.ids[i],
                             proto.proto.prediction[i].id.encode('ascii'))
        self.assertEqual(self.n_predictors, proto.proto.n_predictors)
        self.assertEqual(
            self.n_prediction_dimensions,
            proto.proto.n_prediction_dimensions)
        self.assertEqual(self.predictions.dtype, proto.proto.dtype)
        self.assertEqual('test', proto.proto.predictor)
        self.assertEqual('path', proto.proto.db_path)
        self.assertEqual('ManagedHDF5Database', proto.proto.db_class)


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
        self.labels.db_class = 'ManagedHDF5Database'

        self.tempdir = tempfile.TemporaryDirectory()
        self.path = os.path.join(self.tempdir.name, 'predin.proto')
        self.labels.db_path = os.path.join(self.tempdir.name, 'test.h5')
        with open(self.path, 'wb') as f:
            f.write(self.labels.SerializeToString())

        self.n_instances = 2
        self.features = numpy.array([2, 5, 3, 7]).reshape((self.n_instances, 2))
        with acton.database.ManagedHDF5Database(
                self.labels.db_path, label_dtype=self.labels.dtype,
                feature_dtype='int32') as db:
            db.write_features([self.instance_1.id, self.instance_2.id],
                              self.features)
            labels = numpy.array([i.label for i in self.labels.instance]
                ).reshape((1, -1, 1))
            db.write_labels([b'0'], [self.instance_1.id, self.instance_2.id],
                            labels)

    def tearDown(self):
        self.tempdir.cleanup()

    def testAll(self):
        """Committee can be used with PredictorInput."""
        pred_input = acton.proto.wrappers.PredictorInput(self.labels)
        with pred_input.DB() as db:
            lrc = acton.predictors.Committee(
                acton.predictors.from_class(
                    sklearn.linear_model.LogisticRegression), db,
                n_classifiers=10)
            ids = pred_input.ids
            lrc.fit(ids)
            probs = lrc.predict(ids)
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
        self.labels.db_class = 'ManagedHDF5Database'

        self.tempdir = tempfile.TemporaryDirectory()
        self.path = os.path.join(self.tempdir.name, 'predin.proto')
        self.labels.db_path = os.path.join(self.tempdir.name, 'test.h5')
        with open(self.path, 'wb') as f:
            f.write(self.labels.SerializeToString())

        self.n_instances = 2
        self.features = numpy.array([2, 5, 3, 7]).reshape((self.n_instances, 2))
        with acton.database.ManagedHDF5Database(
                self.labels.db_path, label_dtype=self.labels.dtype,
                feature_dtype='int32') as db:
            db.write_features([self.instance_1.id, self.instance_2.id],
                              self.features)
            labels = numpy.array([i.label for i in self.labels.instance]
                ).reshape((1, -1, 1))
            db.write_labels([b'0'], [self.instance_1.id, self.instance_2.id],
                            labels)

    def tearDown(self):
        self.tempdir.cleanup()

    def testFromInstance(self):
        """from_instance wraps a scikit-learn classifier."""
        # The main point of this test is to check nothing crashes.
        classifier = sklearn.linear_model.LogisticRegression()
        pred_input = acton.proto.wrappers.PredictorInput(self.labels)

        with pred_input.DB() as db:
            predictor = acton.predictors.from_instance(classifier, db)
            ids = pred_input.ids
            predictor.fit(ids)
            probs = predictor.predict(ids)
            self.assertEqual((2, 1), probs.shape)

    def testFromClass(self):
        """from_class wraps a scikit-learn classifier."""
        # The main point of this test is to check nothing crashes.
        Classifier = sklearn.linear_model.LogisticRegression
        pred_input = acton.proto.wrappers.PredictorInput(self.labels)

        with pred_input.DB() as db:
            Predictor = acton.predictors.from_class(Classifier)
            predictor = Predictor(db, C=50.0)
            ids = pred_input.ids
            predictor.fit(ids)
            probs = predictor.predict(ids)
            self.assertEqual((2, 1), probs.shape)
