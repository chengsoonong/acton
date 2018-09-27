#!/usr/bin/env python3

"""
test_database
----------------------------------

Tests for `database` module.
"""

import os.path
import tempfile
import unittest

from acton import database
import numpy


class TestManagedHDF5Database(unittest.TestCase):
    """Tests the ManagedHDF5Database class."""

    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.tempdir.cleanup()

    def temp_path(self, filename: str) -> str:
        """Makes a temporary path for a file.

        Parameters
        ----------
        filename
            Filename of the file.

        Returns
        -------
        str
            Full path to the temporary filename.
        """
        return os.path.join(self.tempdir.name, filename)

    def test_io_features(self):
        """Features can be written to and read out from a ManagedHDF5Database.
        """
        path = self.temp_path('test_io_features.h5')
        # Make some testing data.
        n_instances = 20
        n_dimensions = 15
        ids = [i for i in range(n_instances)]
        numpy.random.shuffle(ids)
        features = numpy.random.random(size=(n_instances, n_dimensions)).astype(
            'float32')

        # Store half the testing data in the database.
        with database.ManagedHDF5Database(path) as db:
            db.write_features(ids[:n_instances // 2],
                              features[:n_instances // 2])
        with database.ManagedHDF5Database(path) as db:
            self.assertTrue(numpy.allclose(
                features[:n_instances // 2],
                db.read_features(ids[:n_instances // 2])))

        # Change some features.
        changed_ids_int = [i for i in range(0, n_instances, 2)]
        changed_ids = [ids[i] for i in changed_ids_int]
        new_features = features.copy()
        new_features[changed_ids_int] = numpy.random.random(
            size=(len(changed_ids), n_dimensions))

        # Update the database and extend it by including all features.
        with database.ManagedHDF5Database(path) as db:
            db.write_features(ids, new_features)
        with database.ManagedHDF5Database(path) as db:
            self.assertTrue(numpy.allclose(
                new_features,
                db.read_features(ids)))

    def test_read_write_labels(self):
        """Labels can be written to and read from a ManagedHDF5Database."""
        path = self.temp_path('test_read_write_labels.h5')
        # Make some testing data.
        n_instances = 5
        n_dimensions = 1
        n_labellers = 1
        ids = [i for i in range(n_instances)]
        labeller_ids = [i for i in range(n_labellers)]
        numpy.random.shuffle(ids)
        labels = numpy.random.random(
            size=(n_labellers, n_instances, n_dimensions)).astype('float32')
        # Store half the testing data in the database.
        with database.ManagedHDF5Database(path) as db:
            db.write_labels(labeller_ids,
                            ids[:n_instances // 2],
                            labels[:, :n_instances // 2])
        with database.ManagedHDF5Database(path) as db:
            exp_labels = labels[:, :n_instances // 2]
            act_labels = db.read_labels(labeller_ids,
                                        ids[:n_instances // 2])
            self.assertTrue(numpy.allclose(exp_labels, act_labels),
                            msg='delta {}'.format(exp_labels - act_labels))
        # Change some labels.
        changed_ids_int = [i for i in range(0, n_instances, 2)]
        changed_ids = [ids[i] for i in changed_ids_int]
        new_labels = labels.copy()
        new_labels[:, changed_ids_int] = numpy.random.random(
            size=(n_labellers, len(changed_ids), n_dimensions))

        # Update the database and extend it by including all labels.
        with database.ManagedHDF5Database(path) as db:
            db.write_labels(labeller_ids, ids, new_labels)
        with database.ManagedHDF5Database(path) as db:
            self.assertTrue(numpy.allclose(
                new_labels,
                db.read_labels(labeller_ids, ids)))
