#!/usr/bin/env python3

"""
test_database
----------------------------------

Tests for `database` module.
"""

import os.path
import sys
import tempfile
import unittest

from acton import database
import numpy


class TestHDF5Database(unittest.TestCase):
    """Tests the HDF5Database class."""

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

    def test_write_features(self):
        """Features can be written to a new database."""
        path = self.temp_path('test_write_features.h5')
        # Make some testing data.
        n_instances = 20
        n_dimensions = 15
        ids = [str(i).encode('ascii') for i in range(n_instances)]
        numpy.random.shuffle(ids)
        features = numpy.random.random(size=(n_instances, n_dimensions))
        # Store the testing data in the database.
        with database.HDF5Database(path) as db:
            db.write_features(ids, features)
        # Check that the testing data was stored.
        with database.HDF5Database(path) as db:
            self.assertTrue(numpy.allclose(
                db._h5_file['features'].value,
                features))
            self.assertEqual([i for i in db._h5_file['ids']],
                             ids)

    def test_update_features(self):
        """Features in an HDF5Database can be updated."""
        path = self.temp_path('test_update_features.h5')
        # Make some testing data.
        n_instances = 20
        n_dimensions = 15
        ids = [str(i).encode('ascii') for i in range(n_instances)]
        numpy.random.shuffle(ids)
        features = numpy.random.random(size=(n_instances, n_dimensions))
        # Store the testing data in the database.
        with database.HDF5Database(path) as db:
            db.write_features(ids, features)
        # Change some features.
        changed_ids_int = [i for i in range(0, n_instances, 2)]
        changed_ids = [ids[i] for i in changed_ids_int]
        new_features = features.copy()
        new_features[changed_ids_int] = numpy.random.random(
            size=(len(changed_ids), n_dimensions))
        with database.HDF5Database(path) as db:
            db.write_features(changed_ids, new_features[changed_ids_int])
            self.assertTrue(numpy.allclose(
                db._h5_file['features'].value,
                new_features))

    def test_update_and_write_features(self):
        """Features can be updated/written to an HDF5Database simultaneously."""
        path = self.temp_path('test_update_and_write_features.h5')
        # Make some testing data.
        n_instances = 20
        n_dimensions = 15
        ids = [str(i).encode('ascii') for i in range(n_instances)]
        numpy.random.shuffle(ids)
        features = numpy.random.random(size=(n_instances, n_dimensions))
        # Store half the testing data in the database.
        with database.HDF5Database(path) as db:
            db.write_features(ids[:n_instances // 2],
                              features[:n_instances // 2])
        # Change some features.
        changed_ids_int = [i for i in range(0, n_instances, 2)]
        changed_ids = [ids[i] for i in changed_ids_int]
        new_features = features.copy()
        new_features[changed_ids_int] = numpy.random.random(
            size=(len(changed_ids), n_dimensions))

        # Update the database and extend it by including all features.
        with database.HDF5Database(path) as db:
            db.write_features(ids, new_features)
            self.assertTrue(numpy.allclose(
                db._h5_file['features'].value,
                new_features))

    def test_read_features(self):
        """Features can be read from an HDF5Database."""
        path = self.temp_path('test_read_features.h5')
        # Make some testing data.
        n_instances = 20
        n_dimensions = 15
        ids = [str(i).encode('ascii') for i in range(n_instances)]
        numpy.random.shuffle(ids)
        features = numpy.random.random(size=(n_instances, n_dimensions))
        # Store the testing data in the database.
        with database.HDF5Database(path) as db:
            db.write_features(ids, features)
        # Read it back from the database.
        with database.HDF5Database(path) as db:
            read_features = db.read_features(ids)
            self.assertTrue(numpy.allclose(
                features,
                read_features))


    def test_read_features_unordered(self):
        """Features can be read from an HDF5Database independently of ID order.
        """
        path = self.temp_path('test_read_features_unordered.h5')
        # Make some testing data.
        n_instances = 20
        n_dimensions = 15
        ids = [str(i).encode('ascii') for i in range(n_instances)]
        numpy.random.shuffle(ids)
        features = numpy.random.random(size=(n_instances, n_dimensions))
        # Store the testing data in the database.
        with database.HDF5Database(path) as db:
            db.write_features(ids, features)
        # Read it back from the database in random order.
        ids_int = list(range(n_instances))
        numpy.random.shuffle(ids_int)
        ids = [ids[i] for i in ids_int]
        with database.HDF5Database(path) as db:
            read_features = db.read_features(ids)
            self.assertTrue(numpy.allclose(
                features[ids_int],
                read_features))
