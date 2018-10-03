#!/usr/bin/env python3

"""
test_integration
----------------------------------

Integration tests.
"""

import sys
sys.path.append("..")


import os.path
import struct
import unittest
import unittest.mock

import acton.cli
import acton.database
import acton.proto.io
import acton.proto.wrappers
import acton.proto.acton_pb2
from click.testing import CliRunner
import numpy


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
                'passive.pb', acton.proto.acton_pb2.Predictions)

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
                'str.pb', acton.proto.acton_pb2.Predictions)

            protos = list(reader)

            self.assertEqual(
                2, len(protos),
                msg='Expected 2 protobufs; found {}'.format(len(protos)))

    def test_classification_gpc_non_integer_labels(self):
        """Acton handles non-integer labels with a GPClassifier."""
        txt_path = os.path.realpath(
            os.path.join(os.path.dirname(os.path.realpath(__file__)),
                         'data', 'classification_str.txt'))
        with self.runner.isolated_filesystem():
            result = self.runner.invoke(
                acton.cli.main,
                ['--data', txt_path,
                 '-o', 'str.pb',
                 '--recommender', 'RandomRecommender',
                 '--predictor', 'GPC',
                 '--epochs', '2',
                 '--label', 'label'])

            if result.exit_code != 0:
                raise result.exception

            self.assertEqual('', result.output)

            self.assertTrue(os.path.exists('str.pb'))

            reader = acton.proto.io.read_protos(
                'str.pb', acton.proto.acton_pb2.Predictions)

            protos = list(reader)

            self.assertEqual(
                2, len(protos),
                msg='Expected 2 protobufs; found {}'.format(len(protos)))

    def test_classification_uncrt_non_integer_labels(self):
        """Acton handles non-integer labels with uncertainty sampling."""
        txt_path = os.path.realpath(
            os.path.join(os.path.dirname(os.path.realpath(__file__)),
                         'data', 'classification_str.txt'))
        with self.runner.isolated_filesystem():
            result = self.runner.invoke(
                acton.cli.main,
                ['--data', txt_path,
                 '-o', 'str.pb',
                 '--recommender', 'UncertaintyRecommender',
                 '--predictor', 'LogisticRegression',
                 '--epochs', '2',
                 '--label', 'label'])

            if result.exit_code != 0:
                raise result.exception

            self.assertEqual('', result.output)

            self.assertTrue(os.path.exists('str.pb'))

            reader = acton.proto.io.read_protos(
                'str.pb', acton.proto.acton_pb2.Predictions)

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
                'str.pb', acton.proto.acton_pb2.Predictions)

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
                'str.pb', acton.proto.acton_pb2.Predictions)

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
                'passive.pb', acton.proto.acton_pb2.Predictions)

            protos = list(reader)

            self.assertEqual(
                2, len(protos),
                msg='Expected 2 protobufs; found {}'.format(len(protos)))

    def test_classification_passive_fits(self):
        """Acton handles a passive classification task with a FITS table."""
        fits_path = os.path.realpath(
            os.path.join(os.path.dirname(os.path.realpath(__file__)),
                         'data', 'classification.fits'))
        with self.runner.isolated_filesystem():
            result = self.runner.invoke(
                acton.cli.main,
                ['--data', fits_path,
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
                'passive.pb', acton.proto.acton_pb2.Predictions)

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
                'uncertainty.pb', acton.proto.acton_pb2.Predictions)

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
                'qbc.pb', acton.proto.acton_pb2.Predictions)

            protos = list(reader)

            self.assertEqual(
                2, len(protos),
                msg='Expected 2 protobufs; found {}'.format(len(protos)))


class TestComponentCLI(unittest.TestCase):
    """Tests the CLI to the label/predict/recommend components."""

    def setUp(self):
        self.runner = CliRunner()

    def test_label_args(self):
        """acton-label takes arguments and outputs a protobuf."""
        db_path = os.path.realpath(
            os.path.join(os.path.dirname(os.path.realpath(__file__)),
                         'data', 'classification_pandas.h5'))

        with self.runner.isolated_filesystem():
            result = self.runner.invoke(
                acton.cli.label,
                ['--data', db_path,
                 '--label', 'col20',
                 '--feature', 'col10',
                 '--feature', 'col11',
                 '--pandas-key', 'classification'],
                input='1\n2\n3\n\n')

            if result.exit_code != 0:
                raise result.exception

            output = result.output_bytes
            length, = struct.unpack('<Q', output[:8])
            proto = output[8:]
            self.assertEqual(len(proto), length)
            labels = acton.proto.wrappers.LabelPool.deserialise(proto)
            self.assertEqual([1, 2, 3], labels.ids)
            self.assertTrue(labels.proto.db.path.endswith('_pandas.h5'))
            self.assertEqual({
                'feature_cols': ['col10', 'col11'],
                'label_col': 'col20',
                'key': 'classification',
                'encode_labels': True,
            }, labels.db_kwargs)

    def test_label_protobuf(self):
        """acton-label takes and outputs a protobuf."""
        db_path = os.path.realpath(
            os.path.join(os.path.dirname(os.path.realpath(__file__)),
                         'data', 'classification_pandas.h5'))

        with self.runner.isolated_filesystem():
            db_kwargs = {
                'feature_cols': ['col10', 'col11'],
                'label_col': 'col20',
                'key': 'classification',
                'encode_labels': True,
            }
            with acton.database.PandasReader(db_path, **db_kwargs) as db:
                proto = acton.proto.wrappers.Recommendations.make(
                    labelled_ids=[1, 2, 3],
                    recommended_ids=[4],
                    recommender='UncertaintyRecommender',
                    db=db).proto.SerializeToString()
            assert isinstance(proto, bytes)
            length = struct.pack('<Q', len(proto))
            result = self.runner.invoke(
                acton.cli.label,
                input=length + proto)

            if result.exit_code != 0:
                raise result.exception

            output = result.output_bytes
            length, = struct.unpack('<Q', output[:8])
            proto = output[8:]
            self.assertEqual(len(proto), length)
            labels = acton.proto.wrappers.LabelPool.deserialise(proto)
            self.assertEqual([1, 2, 3, 4], labels.ids)
            self.assertTrue(labels.proto.db.path.endswith('_pandas.h5'))
            self.assertEqual({
                'feature_cols': ['col10', 'col11'],
                'label_col': 'col20',
                'key': 'classification',
                'encode_labels': True,
            }, labels.db_kwargs)

    def test_predict(self):
        """acton-predict takes and outputs a protobuf."""
        db_path = os.path.realpath(
            os.path.join(os.path.dirname(os.path.realpath(__file__)),
                         'data', 'classification_pandas.h5'))

        with self.runner.isolated_filesystem():
            db_kwargs = {
                'feature_cols': ['col10', 'col11'],
                'label_col': 'col20',
                'key': 'classification',
                'encode_labels': True,
            }
            with acton.database.PandasReader(db_path, **db_kwargs) as db:
                proto = acton.proto.wrappers.LabelPool.make(
                    ids=[1, 2, 3],
                    db=db)
            proto = proto.proto.SerializeToString()
            assert isinstance(proto, bytes)
            length = struct.pack('<Q', len(proto))
            result = self.runner.invoke(
                acton.cli.predict,
                input=length + proto)

            if result.exit_code != 0:
                raise result.exception

            output = result.output_bytes
            length, = struct.unpack('<Q', output[:8])
            proto = output[8:]
            self.assertEqual(len(proto), length)
            predictions = acton.proto.wrappers.Predictions.deserialise(proto)
            self.assertEqual([1, 2, 3], predictions.labelled_ids)
            self.assertTrue(predictions.proto.db.path.endswith('_pandas.h5'))
            output_db_kwargs = predictions.db_kwargs
            del output_db_kwargs['label_encoder']
            self.assertEqual({
                'feature_cols': ['col10', 'col11'],
                'label_col': 'col20',
                'key': 'classification',
                'encode_labels': True,
            }, output_db_kwargs)

    def test_recommend(self):
        """acton-recommend takes and outputs a protobuf."""
        db_path = os.path.realpath(
            os.path.join(os.path.dirname(os.path.realpath(__file__)),
                         'data', 'classification_pandas.h5'))

        with self.runner.isolated_filesystem():
            predictions = numpy.random.random(size=(1, 2, 1))
            db_kwargs = {
                'feature_cols': ['col10', 'col11'],
                'label_col': 'col20',
                'key': 'classification',
                'encode_labels': True,
            }
            with acton.database.PandasReader(db_path, **db_kwargs) as db:
                proto = acton.proto.wrappers.Predictions.make(
                    labelled_ids=[1, 2, 3],
                    predicted_ids=[4, 5],
                    predictions=predictions,
                    predictor='LogisticRegression',
                    db=db)
            proto = proto.proto.SerializeToString()
            assert isinstance(proto, bytes)
            length = struct.pack('<Q', len(proto))
            result = self.runner.invoke(
                acton.cli.recommend,
                input=length + proto)

            if result.exit_code != 0:
                raise result.exception

            output = result.output_bytes
            length, = struct.unpack('<Q', output[:8])
            proto = output[8:]
            self.assertEqual(len(proto), length)
            recs = acton.proto.wrappers.Recommendations.deserialise(proto)
            self.assertEqual([1, 2, 3], recs.labelled_ids)
            self.assertTrue(recs.proto.db.path.endswith('_pandas.h5'))
            self.assertEqual({
                'feature_cols': ['col10', 'col11'],
                'label_col': 'col20',
                'key': 'classification',
                'encode_labels': True,
            }, recs.db_kwargs)
