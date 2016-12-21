"""Wrapper class for databases."""

from abc import ABC, abstractmethod
from inspect import Traceback
import os.path
import tempfile
from typing import Iterable, List
import warnings

import astropy.io.ascii as io_ascii
import astropy.table
import h5py
import numpy
import pandas


def product(seq: Iterable[int]):
    """Finds the product of a list of ints.

    Arguments
    ---------
    seq
        List of ints.

    Returns
    -------
    int
        Product.
    """
    prod = 1
    for i in seq:
        prod *= i
    return prod


class Database(ABC):
    """Base class for database wrappers."""

    @abstractmethod
    def __enter__(self):
        return self

    @abstractmethod
    def __exit__(self, exc_type: Exception, exc_val: object, exc_tb: Traceback):
        pass

    @abstractmethod
    def read_features(self, ids: Iterable[bytes]) -> numpy.ndarray:
        """Reads feature vectors from the database.

        Parameters
        ----------
        ids
            Iterable of IDs.

        Returns
        -------
        numpy.ndarray
            N x D array of feature vectors.
        """

    @abstractmethod
    def read_labels(self,
                    labeller_ids: Iterable[bytes],
                    instance_ids: Iterable[bytes]) -> numpy.ndarray:
        """Reads label vectors from the database.

        Parameters
        ----------
        labeller_ids
            Iterable of labeller IDs.
        instance_ids
            Iterable of instance IDs.

        Returns
        -------
        numpy.ndarray
            T x N x F array of label vectors.
        """

    @abstractmethod
    def write_features(self, ids: Iterable[bytes], features: numpy.ndarray):
        """Writes feature vectors to the database.

        Parameters
        ----------
        ids
            Iterable of IDs.
        features
            N x D array of feature vectors. The ith row corresponds to the ith
            ID in `ids`.
        """

    @abstractmethod
    def write_labels(self,
                     labeller_ids: Iterable[bytes],
                     instance_ids: Iterable[bytes],
                     labels: numpy.ndarray):
        """Writes label vectors to the database.

        Parameters
        ----------
        labeller_ids
            Iterable of labeller IDs.
        instance_ids
            Iterable of instance IDs.
        labels
            T x N x D array of label vectors. The ith row corresponds to the ith
            labeller ID in `labeller_ids` and the jth column corresponds to the
            jth instance ID in `instance_ids`.
        """

    @abstractmethod
    def get_known_instance_ids(self) -> List[bytes]:
        """Returns a list of known instance IDs.

        Returns
        -------
        List[str]
            A list of known instance IDs.
        """

    @abstractmethod
    def get_known_labeller_ids(self) -> List[bytes]:
        """Returns a list of known labeller IDs.

        Returns
        -------
        List[str]
            A list of known labeller IDs.
        """


class HDF5Database(Database):
    """Database wrapping an HDF5 file as a context manager.

    Attributes
    ----------
    path : str
        Path to HDF5 file.
    _h5_file : h5py.File
        HDF5 file object.
    """

    def __init__(self, path: str):
        self.path = path

    def __enter__(self):
        self._open_hdf5()
        return self

    def __exit__(self, exc_type: Exception, exc_val: object, exc_tb: Traceback):
        self._h5_file.close()
        delattr(self, '_h5_file')

    def _assert_open(self):
        """Asserts that the HDF5 file is ready to be read to/written from.

        Raises
        ------
        AssertionError
        """
        assert hasattr(self, '_h5_file'), ('HDF5 database must be used as a '
                                           'context manager.')

    def _open_hdf5(self):
        """Opens the HDF5 file and creates it if it doesn't exist.

        Notes
        -----
        The HDF5 file will be stored in self._h5_file.
        """
        try:
            self._h5_file = h5py.File(self.path, 'r+')
        except OSError:
            with h5py.File(self.path, 'w') as h5_file:
                self._setup_hdf5(h5_file)
            self._h5_file = h5py.File(self.path, 'r+')


class ManagedHDF5Database(HDF5Database):
    """Database using an HDF5 file.

    Notes
    -----
    This database uses an internal schema. For reading files from disk, use
    another Database.

    Attributes
    ----------
    path : str
        Path to HDF5 file.
    label_dtype : str
        Data type of labels.
    feature_dtype : str
        Data type of features.
    max_id_length : int
        Maximum length of ID strings.
    _h5_file : h5py.File
        Opened HDF5 file.
    _sync_attrs : List[str]
        List of instance attributes to sync with the HDF5 file's attributes.
    """

    def __init__(self, path: str, label_dtype: str=None,
                 feature_dtype: str=None,
                 max_id_length: int=None):
        """
        Parameters
        ----------
        path
            Path to HDF5 file.
        label_dtype
            Data type of labels. If not provided then it will be read from the
            database file; if the database file does not exist then the default
            type of 'float32' will be used.
        feature_dtype
            Data type of features. If not provided then it will be read from the
            database file; if the database file does not exist then the default
            type of 'float32' will be used.
        max_id_length
            Maximum length of the IDs this database will store. If not provided
            then it will be read from the database file; if the database file
            does not exist then a default value of 128 will be used.
        """
        super().__init__(path)
        self.label_dtype = label_dtype
        self._default_label_dtype = 'float32'
        self.feature_dtype = feature_dtype
        self._default_feature_dtype = 'float32'
        self.max_id_length = max_id_length
        self._default_max_id_length = 128

        # List of attributes to keep in sync with the HDF5 file.
        self._sync_attrs = ['max_id_length', 'label_dtype', 'feature_dtype']

    def _open_hdf5(self):
        """Opens the HDF5 file and creates it if it doesn't exist.

        Notes
        -----
        The HDF5 file will be stored in self._h5_file.
        """
        super()._open_hdf5()

        # Load attrs from HDF5 file if we haven't specified them.
        for attr in self._sync_attrs:
            if getattr(self, attr) is None:
                setattr(self, attr, self._h5_file.attrs[attr])

        self._validate_hdf5()

    def write_features(self, ids: Iterable[bytes], features: numpy.ndarray):
        """Writes feature vectors to the database.

        Parameters
        ----------
        ids
            Iterable of IDs.
        features:
            N x D array of feature vectors. The ith row corresponds to the ith
            ID in `ids`.

        Returns
        -------
        numpy.ndarray
            N x D array of feature vectors.
        """
        self._assert_open()

        # Input validation.
        if len(ids) != len(features):
            raise ValueError('Must have same number of IDs and features.')

        if self._h5_file.attrs['n_features'] == -1:
            # This is the first time we've stored features, so make a record of
            # the dimensionality.
            self._h5_file.attrs['n_features'] = features.shape[1]
        elif self._h5_file.attrs['n_features'] != features.shape[1]:
            raise ValueError(
                'Expected features to have dimensionality {}, got {}'.format(
                    self._h5_file.attrs['n_features'], features.shape[1]))

        # Cast the features to the right type.
        if features.dtype != self.feature_dtype:
            warnings.warn('Casting features from type {} to type {}.'.format(
                features.dtype, self.feature_dtype))
            features = features.astype(self.feature_dtype)

        # Store the feature vectors.
        for id_, feature in zip(ids, features):
            if id_ not in self._h5_file['features']:
                self._h5_file['features'].create_dataset(name=id_, data=feature)
            else:
                self._h5_file['features'][id_][:] = feature

        # Add the IDs to the database.
        known_ids = set(self.get_known_instance_ids())
        new_ids = [i for i in ids if i not in known_ids]
        n_new_ids = len(new_ids)
        n_old_ids = self._h5_file['instance_ids'].shape[0]
        self._h5_file['instance_ids'].resize((n_old_ids + n_new_ids,))
        self._h5_file['instance_ids'][-n_new_ids:] = numpy.array(
            new_ids, dtype='<S{}'.format(self.max_id_length))

    def read_features(self, ids: Iterable[bytes]) -> numpy.ndarray:
        """Reads feature vectors from the database.

        Parameters
        ----------
        ids
            Iterable of IDs.

        Returns
        -------
        numpy.ndarray
            N x D array of feature vectors.
        """
        self._assert_open()

        if self._h5_file.attrs['n_features'] == -1 and ids:
            raise KeyError('No features stored in database.')

        # Allocate the features array.
        features = numpy.zeros((len(ids), self._h5_file.attrs['n_features']),
                               dtype=self._h5_file.attrs['feature_dtype'])
        # Loop through each ID we want to query and put the associated feature
        # into the features array.
        for idx, id_ in enumerate(ids):
            try:
                feature = self._h5_file['features'][id_]
            except KeyError:
                raise KeyError('Unknown ID: {}'.format(id_))
            features[idx] = feature
        return features

    def write_labels(self,
                     labeller_ids: Iterable[bytes],
                     instance_ids: Iterable[bytes],
                     labels: numpy.ndarray):
        """Writes label vectors to the database.

        Parameters
        ----------
        labeller_ids
            Iterable of labeller IDs.
        instance_ids
            Iterable of instance IDs.
        labels
            T x N x D array of label vectors. The ith row corresponds to the ith
            labeller ID in `labeller_ids` and the jth column corresponds to the
            jth instance ID in `instance_ids`.
        """
        self._assert_open()

        # Input validation.
        if len(labeller_ids) != labels.shape[0]:
            raise ValueError(
                'labels array has incorrect number of labellers:'
                ' expected {}, got {}.'.format(len(labeller_ids),
                                               labels.shape[0]))

        if len(instance_ids) != labels.shape[1]:
            raise ValueError(
                'labels array has incorrect number of instances:'
                ' expected {}, got {}.'.format(len(instance_ids),
                                               labels.shape[1]))

        if self._h5_file.attrs['label_dim'] == -1:
            # This is the first time we've stored labels, so make a record of
            # the dimensionality.
            self._h5_file.attrs['label_dim'] = labels.shape[2]
        elif self._h5_file.attrs['label_dim'] != labels.shape[2]:
            raise ValueError(
                'Expected labels to have dimensionality {}, got {}'.format(
                    self._h5_file.attrs['label_dim'], labels.shape[2]))

        # Cast the labels to the right type.
        if labels.dtype != self.label_dtype:
            warnings.warn('Casting labels from type {} to type {}.'.format(
                labels.dtype, self.label_dtype))
            labels = labels.astype(self.label_dtype)

        # Store the labels.
        for labeller_idx, labeller_id in enumerate(labeller_ids):
            for instance_idx, instance_id in enumerate(instance_ids):
                if labeller_id not in self._h5_file['labels']:
                    labeller = self._h5_file['labels'].create_group(
                        name=labeller_id)
                    labeller.create_dataset(
                        name=instance_id,
                        data=labels[labeller_idx, instance_idx])
                elif instance_id not in self._h5_file['labels'][labeller_id]:
                    self._h5_file['labels'][labeller_id].create_dataset(
                        name=instance_id,
                        data=labels[labeller_idx, instance_idx])
                else:
                    self._h5_file['labels'][labeller_id][instance_idx][:] = (
                        labels[labeller_idx, instance_idx])

        # Add the instance IDs to the database.
        known_instance_ids = set(self.get_known_instance_ids())
        new_instance_ids = [i for i in instance_ids
                            if i not in known_instance_ids]
        n_new_instance_ids = len(new_instance_ids)
        n_old_instance_ids = self._h5_file['instance_ids'].shape[0]
        if n_new_instance_ids:
            self._h5_file['instance_ids'].resize(
                (n_old_instance_ids + n_new_instance_ids,))
            self._h5_file['instance_ids'][-n_new_instance_ids:] = numpy.array(
                new_instance_ids, dtype='<S{}'.format(self.max_id_length))

        # Add the labeller IDs to the database.
        known_labeller_ids = set(self.get_known_labeller_ids())
        new_labeller_ids = [i for i in labeller_ids
                            if i not in known_labeller_ids]
        n_new_labeller_ids = len(new_labeller_ids)
        n_old_labeller_ids = self._h5_file['labeller_ids'].shape[0]
        if n_new_labeller_ids:
            self._h5_file['labeller_ids'].resize(
                (n_old_labeller_ids + n_new_labeller_ids,))
            self._h5_file['labeller_ids'][-n_new_labeller_ids:] = numpy.array(
                new_labeller_ids, dtype='<S{}'.format(self.max_id_length))

    def read_labels(self,
                    labeller_ids: Iterable[bytes],
                    instance_ids: Iterable[bytes]) -> numpy.ndarray:
        """Reads label vectors from the database.

        Parameters
        ----------
        labeller_ids
            Iterable of labeller IDs.
        instance_ids
            Iterable of instance IDs.

        Returns
        -------
        numpy.ndarray
            T x N x F array of label vectors.
        """
        self._assert_open()

        if self._h5_file.attrs['label_dim'] == -1 and (
                labeller_ids or instance_ids):
            raise KeyError('No labels stored in database.')

        # Allocate the labels array.
        labels = numpy.zeros(
            (len(labeller_ids),
             len(instance_ids),
             self._h5_file.attrs['label_dim']),
            dtype=self._h5_file.attrs['label_dtype'])
        # Loop through each ID we want to query and put the associated label
        # into the labels array.
        for labeller_idx, labeller_id in enumerate(labeller_ids):
            for instance_idx, instance_id in enumerate(instance_ids):
                try:
                    labeller_labels = self._h5_file['labels'][labeller_id]
                except KeyError:
                    raise KeyError(
                        'Unknown labeller ID: {}'.format(labeller_id))

                try:
                    label = labeller_labels[instance_id]
                except KeyError:
                    raise KeyError(
                        'Unknown instance ID: {}'.format(instance_id))

                labels[labeller_idx, instance_idx] = label

        return labels

    def get_known_instance_ids(self) -> List[bytes]:
        """Returns a list of known instance IDs.

        Returns
        -------
        List[str]
            A list of known instance IDs.
        """
        self._assert_open()
        return [id_ for id_ in self._h5_file['instance_ids']]

    def get_known_labeller_ids(self) -> List[bytes]:
        """Returns a list of known labeller IDs.

        Returns
        -------
        List[str]
            A list of known labeller IDs.
        """
        self._assert_open()
        return [id_ for id_ in self._h5_file['labeller_ids']]

    def _setup_hdf5(self, h5_file: h5py.File):
        """Sets up an HDF5 file to work as a database.

        Parameters
        ----------
        h5_file
            HDF5 file to set up. Must be opened in write mode.
        """
        if self.max_id_length is None:
            self.max_id_length = self._default_max_id_length
        if self.label_dtype is None:
            self.label_dtype = self._default_label_dtype
        if self.feature_dtype is None:
            self.feature_dtype = self._default_feature_dtype
        h5_file.create_group('features')
        h5_file.create_group('labels')
        h5_file.create_dataset('instance_ids', shape=(0,),
                               dtype='<S{}'.format(self.max_id_length),
                               maxshape=(None,))
        h5_file.create_dataset('labeller_ids', shape=(0,),
                               dtype='<S{}'.format(self.max_id_length),
                               maxshape=(None,))
        h5_file.attrs['max_id_length'] = self.max_id_length
        h5_file.attrs['label_dtype'] = self.label_dtype
        h5_file.attrs['feature_dtype'] = self.feature_dtype
        h5_file.attrs['n_features'] = -1
        h5_file.attrs['label_dim'] = -1

    def _validate_hdf5(self):
        """Checks that self._h5_file has the correct schema.

        Raises
        ------
        ValueError
        """
        try:
            assert 'features' in self._h5_file
            assert 'labels' in self._h5_file
            assert 'instance_ids' in self._h5_file
            assert 'labeller_ids' in self._h5_file
        except AssertionError:
            raise ValueError(
                'File {} is not a valid database.'.format(self.path))

        for attr in self._sync_attrs:
            assert getattr(self, attr) is not None
            if self._h5_file.attrs[attr] != getattr(self, attr):
                raise ValueError('Incompatible {}: expected {}, got {}'.format(
                    attr, getattr(self, attr), self._h5_file.attrs[attr]))


class HDF5Reader(HDF5Database):
    """Reads HDF5 databases.

    Attributes
    ----------
    feature_cols : List[str]
        List of feature datasets.
    id_col : str
        Name of ID dataset.
    label_col : str
        Name of label dataset.
    n_features : int
        Number of features.
    n_instances : int
        Number of instances.
    n_labels : int
        Number of labels per instance.
    path : str
        Path to HDF5 file.
    _h5_file : h5py.File
        HDF5 file object.
    _is_multidimensional : bool
        Whether the features are in a multidimensional dataset.
    """

    def __init__(self, path: str, feature_cols: List[str], label_col: str,
                 id_col: str=None):
        """
        Parameters
        ----------
        path
            Path to HDF5 file.
        feature_cols
            List of feature datasets. If only one feature dataset is specified,
            this dataset is allowed to be a multidimensional dataset and contain
            multiple features.
        label_col
            Name of label dataset.
        id_col
            Name of ID dataset.
        """
        super().__init__(path)

        if not feature_cols:
            raise ValueError('Must specify feature columns for HDF5.')

        self.feature_cols = feature_cols
        self.label_col = label_col
        self.id_col = id_col
        with h5py.File(self.path, 'r') as data:
            is_multidimensional = any(len(data[f_col].shape) > 1 or
                                      not product(data[f_col].shape[1:]) == 1
                                      for f_col in feature_cols)
            if is_multidimensional and len(feature_cols) != 1:
                raise ValueError(
                    'Feature arrays and feature columns cannot be mixed. '
                    'To read in features from a multidimensional dataset, '
                    'only specify one feature column name.')

            self._is_multidimensional = is_multidimensional

            self.n_instances = data[label_col].shape[0]
            if len(data[label_col].shape) == 1:
                self.n_labels = 1
            else:
                assert len(data[label_col].shape) == 2
                self.n_labels = data[label_col].shape[1]

            if is_multidimensional:
                self.n_features = data[feature_cols[0]].shape[1]
            else:
                self.n_features = len(feature_cols)

    def read_features(self, ids: Iterable[bytes]) -> numpy.ndarray:
        """Reads feature vectors from the database.

        Parameters
        ----------
        ids
            Iterable of IDs.

        Returns
        -------
        numpy.ndarray
            N x D array of feature vectors.
        """
        # TODO(MatthewJA): Optimise this.
        self._assert_open()
        # Allocate output features array.
        features = numpy.zeros((len(ids), self.n_features))
        # For each ID, get the corresponding features. This could be achieved
        # significantly faster.
        for out_index, id_ in enumerate(ids):
            for in_index, id__ in enumerate(self.get_known_instance_ids()):
                if id_ != id__:
                    continue

                if self._is_multidimensional:
                    instance_features = self._h5_file[
                        self.feature_cols[0]][in_index]
                    features[out_index] = instance_features
                else:
                    for feature_index, feature_col in enumerate(
                            self.feature_cols):
                        features[out_index, feature_index] = self._h5_file[
                            feature_col
                        ][in_index]
                break
            else:
                raise ValueError('Unknown ID: {}'.format(id_))
        return numpy.nan_to_num(features)

    def read_labels(self,
                    labeller_ids: Iterable[bytes],
                    instance_ids: Iterable[bytes]) -> numpy.ndarray:
        """Reads label vectors from the database.

        Parameters
        ----------
        labeller_ids
            Iterable of labeller IDs.
        instance_ids
            Iterable of instance IDs.

        Returns
        -------
        numpy.ndarray
            T x N x F array of label vectors.
        """
        # TODO(MatthewJA): Optimise this.
        self._assert_open()
        # Allocate output labels array.
        labels = numpy.zeros(
            (len(labeller_ids), len(instance_ids), self.n_labels))

        if len(labeller_ids) > 1:
            raise NotImplementedError('Multiple labellers not yet supported.')

        # For each ID, get the corresponding labels. This could be achieved
        # significantly faster.
        for out_index, id_ in enumerate(instance_ids):
            for in_index, id__ in enumerate(self.get_known_instance_ids()):
                if id_ != id__:
                    continue

                instance_labels = self._h5_file[
                    self.label_col][in_index]
                labels[0, out_index] = instance_labels
                break
            else:
                raise ValueError('Unknown ID: {}'.format(id_))

        return labels

    def write_features(self, ids: Iterable[bytes], features: numpy.ndarray):
        raise PermissionError('Cannot write to read-only database.')

    def write_labels(self,
                     labeller_ids: Iterable[bytes],
                     instance_ids: Iterable[bytes],
                     labels: numpy.ndarray):
        raise PermissionError('Cannot write to read-only database.')

    def get_known_instance_ids(self) -> List[bytes]:
        """Returns a list of known instance IDs.

        Returns
        -------
        List[str]
            A list of known instance IDs.
        """
        self._assert_open()
        if not self.id_col:
            return [str(i).encode('utf-8') for i in range(self.n_instances)]
        else:
            return list(self._h5_file[self.id_col])

    def get_known_labeller_ids(self) -> List[bytes]:
        """Returns a list of known labeller IDs.

        Returns
        -------
        List[str]
            A list of known labeller IDs.
        """
        raise NotImplementedError()


class ASCIIReader(Database):
    """Reads ASCII databases.

    Attributes
    ----------
    feature_cols : List[str]
        List of feature columns.
    id_col : str
        Name of ID column.
    label_col : str
        Name of label column.
    max_id_length : int
        Maximum length of IDs.
    n_features : int
        Number of features.
    n_instances : int
        Number of instances.
    n_labels : int
        Number of labels per instance.
    path : str
        Path to ASCII file.
    _db : Database
        Underlying ManagedHDF5Database.
    _db_filepath : str
        Path of underlying HDF5 database.
    _tempdir : str
        Temporary directory where the underlying HDF5 database is stored.
    """

    def __init__(self, path: str, feature_cols: List[str], label_col: str,
                 id_col: str=None):
        """
        Parameters
        ----------
        path
            Path to ASCII file.
        feature_cols
            List of feature columns.
        label_col
            Name of label column.
        id_col
            Name of ID column.
        """
        self.path = path
        self.feature_cols = feature_cols
        self.label_col = label_col
        self.id_col = id_col

    @staticmethod
    def _db_from_ascii(
            db: Database,
            data: astropy.table.Table,
            feature_cols: List[str],
            label_col: str,
            ids: List[bytes],
            id_col: str=None):
        """Reads an ASCII table into a database.

        Notes
        -----
        The entire file is copied into memory.

        Arguments
        ---------
        db
            Database.
        data
            ASCII table.
        feature_cols
            List of column names of the features. If empty, all non-label and
            non-ID columns will be used.
        label_col
            Column name of the labels.
        id_col
            Column name of the IDs.
        ids
            List of instance IDs.
        """
        # Read in features.
        columns = data.keys()
        if not feature_cols:
            # If there are no features given, use all columns.
            feature_cols = [c for c in columns
                            if c != label_col and c != id_col]

        # This converts the features from a table to an array.
        features = data[feature_cols].as_array()
        features = features.view(numpy.float64).reshape(features.shape + (-1,))

        # Read in labels.
        labels = numpy.array(data[label_col], dtype=bool).reshape((1, -1, 1))

        # We want to support multiple labellers in the future, but currently
        # don't. So every labeller is the same, ID = 0.
        labeller_ids = [b'0']

        # Write to database.
        db.write_features(ids, features)
        db.write_labels(labeller_ids, ids, labels)

    def __enter__(self):
        self._tempdir = tempfile.TemporaryDirectory(prefix='acton')
        # Read the whole file into a DB.
        self._db_filepath = os.path.join(self._tempdir.name, 'db.h5')
        # First, find the maximum ID length. Do this by reading in IDs.
        data = io_ascii.read(self.path)
        if self.id_col:
            ids = [str(id_).encode('utf-8') for id_ in data[self.id_col]]
        else:
            ids = [str(id_).encode('utf-8')
                   for id_ in range(len(data[self.label_col]))]

        self.max_id_length = max(len(id_) for id_ in ids)

        self._db = ManagedHDF5Database(
            self._db_filepath,
            max_id_length=self.max_id_length,
            label_dtype='bool',
            feature_dtype='float64')
        self._db.__enter__()
        self._db_from_ascii(self._db, data, self.feature_cols, self.label_col,
                            ids, self.id_col)
        return self

    def __exit__(self, exc_type: Exception, exc_val: object, exc_tb: Traceback):
        self._db.__exit__(exc_type, exc_val, exc_tb)
        self._tempdir.cleanup()
        delattr(self, '_db')

    def read_features(self, ids: Iterable[bytes]) -> numpy.ndarray:
        """Reads feature vectors from the database.

        Parameters
        ----------
        ids
            Iterable of IDs.

        Returns
        -------
        numpy.ndarray
            N x D array of feature vectors.
        """
        return self._db.read_features(ids)

    def read_labels(self,
                    labeller_ids: Iterable[bytes],
                    instance_ids: Iterable[bytes]) -> numpy.ndarray:
        """Reads label vectors from the database.

        Parameters
        ----------
        labeller_ids
            Iterable of labeller IDs.
        instance_ids
            Iterable of instance IDs.

        Returns
        -------
        numpy.ndarray
            T x N x F array of label vectors.
        """
        return self._db.read_labels(labeller_ids, instance_ids)

    def write_features(self, ids: Iterable[bytes], features: numpy.ndarray):
        raise NotImplementedError('Cannot write to read-only database.')

    def write_labels(self,
                     labeller_ids: Iterable[bytes],
                     instance_ids: Iterable[bytes],
                     labels: numpy.ndarray):
        raise NotImplementedError('Cannot write to read-only database.')

    def get_known_instance_ids(self) -> List[bytes]:
        """Returns a list of known instance IDs.

        Returns
        -------
        List[str]
            A list of known instance IDs.
        """
        return self._db.get_known_instance_ids()

    def get_known_labeller_ids(self) -> List[bytes]:
        """Returns a list of known labeller IDs.

        Returns
        -------
        List[str]
            A list of known labeller IDs.
        """
        return self._db.get_known_labeller_ids()


class PandasReader(Database):
    """Reads HDF5 databases.

    Attributes
    ----------
    feature_cols : List[str]
        List of feature datasets.
    id_col : str
        Name of ID dataset.
    label_col : str
        Name of label dataset.
    n_features : int
        Number of features.
    n_instances : int
        Number of instances.
    n_labels : int
        Number of labels per instance.
    path : str
        Path to HDF5 file.
    _df : pandas.DataFrame
        Pandas dataframe.
    """

    def __init__(self, path: str, feature_cols: List[str], label_col: str,
                 key: str, id_col: str=None):
        """
        Parameters
        ----------
        path
            Path to HDF5 file.
        feature_cols
            List of feature columns. If none are specified, then all non-label,
            non-ID columns will be used.
        label_col
            Name of label dataset.
        key
            Pandas key.
        id_col
            Name of ID dataset.
        """
        self.path = path
        self.feature_cols = feature_cols
        self.label_col = label_col
        self.id_col = id_col
        self.key = key
        self._df = pandas.read_hdf(self.path, self.key)
        if self.id_col:
            # Build an index for quickly querying the ID column.
            self._df.set_index([self.id_col])

        if not self.feature_cols:
            self.feature_cols = [k for k in self._df.keys()
                                 if k not in {self.label_col, self.id_col}]

        self.n_instances = len(self._df[self.label_col])
        self.n_features = len(self.feature_cols)

    def __enter__(self):
        return self

    def __exit__(self, exc_type: Exception, exc_val: object, exc_tb: Traceback):
        delattr(self, '_df')

    def read_features(self, ids: Iterable[bytes]) -> numpy.ndarray:
        """Reads feature vectors from the database.

        Parameters
        ----------
        ids
            Iterable of IDs.

        Returns
        -------
        numpy.ndarray
            N x D array of feature vectors.
        """
        # TODO(MatthewJA): Optimise this.
        # Allocate output features array.
        features = numpy.zeros((len(ids), self.n_features))
        # For each ID, get the corresponding features.
        for out_index, id_ in enumerate(ids):
            if self.id_col:
                sel = self._df.loc[self._df[self.id_col] == id_]
                if not sel:
                    raise ValueError('Unknown ID: {}'.format(id_))
                sel = sel.ix[0]
            else:
                sel = self._df.ix[int(id_)]

            for feature_index, feature in enumerate(self.feature_cols):
                features[out_index, feature_index] = sel[feature]

        return features

    def read_labels(self,
                    labeller_ids: Iterable[bytes],
                    instance_ids: Iterable[bytes]) -> numpy.ndarray:
        """Reads label vectors from the database.

        Parameters
        ----------
        labeller_ids
            Iterable of labeller IDs.
        instance_ids
            Iterable of instance IDs.

        Returns
        -------
        numpy.ndarray
            T x N x 1 array of label vectors.
        """
        # Allocate output labels array.
        labels = numpy.zeros(
            (len(labeller_ids), len(instance_ids), 1))

        if len(labeller_ids) > 1:
            raise NotImplementedError('Multiple labellers not yet supported.')

        # For each ID, get the corresponding labels.
        for out_index, id_ in enumerate(instance_ids):
            if self.id_col:
                sel = self._df.loc[self._df[self.id_col] == id_]
                if not sel:
                    raise ValueError('Unknown ID: {}'.format(id_))
                sel = sel.ix[0]
            else:
                sel = self._df.ix[int(id_)]

            labels[0, out_index, 0] = sel[self.label_col]

        return labels

    def write_features(self, ids: Iterable[bytes], features: numpy.ndarray):
        raise PermissionError('Cannot write to read-only database.')

    def write_labels(self,
                     labeller_ids: Iterable[bytes],
                     instance_ids: Iterable[bytes],
                     labels: numpy.ndarray):
        raise PermissionError('Cannot write to read-only database.')

    def get_known_instance_ids(self) -> List[bytes]:
        """Returns a list of known instance IDs.

        Returns
        -------
        List[str]
            A list of known instance IDs.
        """
        if not self.id_col:
            return [str(i).encode('utf-8') for i in range(self.n_instances)]
        else:
            return list(self._df[self.id_col])

    def get_known_labeller_ids(self) -> List[bytes]:
        """Returns a list of known labeller IDs.

        Returns
        -------
        List[str]
            A list of known labeller IDs.
        """
        raise NotImplementedError()


# For safe string-based access to database classes.
DATABASES = {
    'ASCIIReader': ASCIIReader,
    'HDF5Reader': HDF5Reader,
    'ManagedHDF5Database': ManagedHDF5Database,
    'PandasReader': PandasReader,
}
