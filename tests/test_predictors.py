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

from acton import predictors
from acton.proto.predictors_pb2 import Labels
import numpy


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
        self.tempdir = tempfile.TemporaryDirectory()
        self.path = os.path.join(self.tempdir.name, 'predin.proto')
        with open(self.path, 'wb') as f:
            f.write(self.labels.SerializeToString())

    def tearDown(self):
        self.tempdir.cleanup()

    def test_init(self):
        """PredictorInput reads a .proto file."""
        predictor_input = predictors.PredictorInput(self.path)
        self.assertEqual('float32', predictor_input.proto.dtype)

    def test_labels(self):
        """PredictorInput gives a label array."""
        predictor_input = predictors.PredictorInput(self.path)
        self.assertTrue(numpy.all(
            predictor_input.labels == numpy.array([[[0]], [[1]]])))  # T x N x F
        self.assertTrue(numpy.all(
            predictor_input.labels == numpy.array([[[0]], [[1]]])))  # T x N x F
        # Second assertion because the property is cached after the first run.
