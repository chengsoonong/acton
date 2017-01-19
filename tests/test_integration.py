#!/usr/bin/env python3

"""
test_integration
----------------------------------

Integration tests.
"""

import os.path
import unittest

import acton.cli
import acton.proto.io
import acton.proto.predictors_pb2
from click.testing import CliRunner


class TestIntegration(unittest.TestCase):

    def setUp(self):
        self.runner = CliRunner()

    def tearDown(self):
        pass

    def test_classification_passive_txt(self):
        """Acton handles a passive classification task with an ASCII file."""
        txt_path = os.path.realpath(
            os.path.join(os.path.dirname(os.path.realpath(__file__)),
                         'data', 'classification.txt'))
        with self.runner.isolated_filesystem():
            result = self.runner.invoke(
                acton.cli.main,
                ['--data', txt_path,
                 '-o', 'passive.pb',
                 '--recommender', 'RandomRecommender',
                 '--predictor', 'LogisticRegression',
                 '--epochs', '2',
                 '--label', 'col20'])

            if result.exit_code != 0:
                raise result.exception

            self.assertEqual('', result.output)

            self.assertTrue(os.path.exists('passive.pb'))

            reader = acton.proto.io.read_protos(
                'passive.pb', acton.proto.predictors_pb2.Predictions)

            protos = list(reader)

            self.assertEqual(
                2, len(protos),
                msg='Expected 2 protobufs; found {}'.format(len(protos)))

    def test_classification_ascii_non_integer_labels(self):
        """Acton handles non-integer labels in an ASCII table."""
        txt_path = os.path.realpath(
            os.path.join(os.path.dirname(os.path.realpath(__file__)),
                         'data', 'classification_str.txt'))
        with self.runner.isolated_filesystem():
            result = self.runner.invoke(
                acton.cli.main,
                ['--data', txt_path,
                 '-o', 'str.pb',
                 '--recommender', 'RandomRecommender',
                 '--predictor', 'LogisticRegression',
                 '--epochs', '2',
                 '--label', 'label'])

            if result.exit_code != 0:
                raise result.exception

            self.assertEqual('', result.output)

            self.assertTrue(os.path.exists('str.pb'))

            reader = acton.proto.io.read_protos(
                'str.pb', acton.proto.predictors_pb2.Predictions)

            protos = list(reader)

            self.assertEqual(
                2, len(protos),
                msg='Expected 2 protobufs; found {}'.format(len(protos)))

    def test_classification_pandas_non_integer_labels(self):
        """Acton handles non-integer labels in a pandas table."""
        txt_path = os.path.realpath(
            os.path.join(os.path.dirname(os.path.realpath(__file__)),
                         'data', 'classification_pandas_str.h5'))
        with self.runner.isolated_filesystem():
            result = self.runner.invoke(
                acton.cli.main,
                ['--data', txt_path,
                 '-o', 'str.pb',
                 '--recommender', 'RandomRecommender',
                 '--predictor', 'LogisticRegression',
                 '--epochs', '2',
                 '--label', 'label',
                 '--pandas-key', 'classification'])

            if result.exit_code != 0:
                raise result.exception

            self.assertEqual('', result.output)

            self.assertTrue(os.path.exists('str.pb'))

            reader = acton.proto.io.read_protos(
                'str.pb', acton.proto.predictors_pb2.Predictions)

            protos = list(reader)

            self.assertEqual(
                2, len(protos),
                msg='Expected 2 protobufs; found {}'.format(len(protos)))

    def test_classification_h5_non_integer_labels(self):
        """Acton handles non-integer labels in an HDF5 table."""
        txt_path = os.path.realpath(
            os.path.join(os.path.dirname(os.path.realpath(__file__)),
                         'data', 'classification_str.h5'))
        with self.runner.isolated_filesystem():
            result = self.runner.invoke(
                acton.cli.main,
                ['--data', txt_path,
                 '-o', 'str.pb',
                 '--recommender', 'RandomRecommender',
                 '--predictor', 'LogisticRegression',
                 '--epochs', '2',
                 '--label', 'labels',
                 '--feature', 'features'])

            if result.exit_code != 0:
                raise result.exception

            self.assertEqual('', result.output)

            self.assertTrue(os.path.exists('str.pb'))

            reader = acton.proto.io.read_protos(
                'str.pb', acton.proto.predictors_pb2.Predictions)

            protos = list(reader)

            self.assertEqual(
                2, len(protos),
                msg='Expected 2 protobufs; found {}'.format(len(protos)))

    def test_classification_passive_pandas(self):
        """Acton handles a passive classification task with a pandas table."""
        pandas_path = os.path.realpath(
            os.path.join(os.path.dirname(os.path.realpath(__file__)),
                         'data', 'classification_pandas.h5'))
        with self.runner.isolated_filesystem():
            result = self.runner.invoke(
                acton.cli.main,
                ['--data', pandas_path,
                 '-o', 'passive.pb',
                 '--recommender', 'RandomRecommender',
                 '--predictor', 'LogisticRegression',
                 '--epochs', '2',
                 '--label', 'col20',
                 '--pandas-key', 'classification'])

            if result.exit_code != 0:
                raise result.exception

            self.assertEqual('', result.output)

            self.assertTrue(os.path.exists('passive.pb'))

            reader = acton.proto.io.read_protos(
                'passive.pb', acton.proto.predictors_pb2.Predictions)

            protos = list(reader)

            self.assertEqual(
                2, len(protos),
                msg='Expected 2 protobufs; found {}'.format(len(protos)))

    def test_classification_uncertainty(self):
        """Acton handles a classification task with uncertainty sampling."""
        txt_path = os.path.realpath(
            os.path.join(os.path.dirname(os.path.realpath(__file__)),
                         'data', 'classification.txt'))
        with self.runner.isolated_filesystem():
            result = self.runner.invoke(
                acton.cli.main,
                ['--data', txt_path,
                 '-o', 'uncertainty.pb',
                 '--recommender', 'UncertaintyRecommender',
                 '--predictor', 'LogisticRegression',
                 '--epochs', '2',
                 '--label', 'col20'])

            if result.exit_code != 0:
                raise result.exception

            self.assertEqual('', result.output)

            self.assertTrue(os.path.exists('uncertainty.pb'))

            reader = acton.proto.io.read_protos(
                'uncertainty.pb', acton.proto.predictors_pb2.Predictions)

            protos = list(reader)

            self.assertEqual(
                2, len(protos),
                msg='Expected 2 protobufs; found {}'.format(len(protos)))

    def test_classification_qbc(self):
        """Acton handles a classification task with QBC and a LR committee."""
        txt_path = os.path.realpath(
            os.path.join(os.path.dirname(os.path.realpath(__file__)),
                         'data', 'classification.txt'))
        with self.runner.isolated_filesystem():
            result = self.runner.invoke(
                acton.cli.main,
                ['--data', txt_path,
                 '-o', 'qbc.pb',
                 '--recommender', 'QBCRecommender',
                 '--predictor', 'LogisticRegressionCommittee',
                 '--epochs', '2',
                 '--label', 'col20'])

            if result.exit_code != 0:
                raise result.exception

            self.assertEqual('', result.output)

            self.assertTrue(os.path.exists('qbc.pb'))

            reader = acton.proto.io.read_protos(
                'qbc.pb', acton.proto.predictors_pb2.Predictions)

            protos = list(reader)

            self.assertEqual(
                2, len(protos),
                msg='Expected 2 protobufs; found {}'.format(len(protos)))
