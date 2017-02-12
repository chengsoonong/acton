"""Wrapper class for databases."""

from abc import ABC, abstractmethod
from inspect import Traceback
import json
import logging
import os.path
import tempfile
from typing import Iterable, List, Sequence
import warnings

from acton.proto.acton_pb2 import Database as DatabasePB
import astropy.io.ascii as io_ascii
import astropy.io.fits as io_fits
import astropy.table
import h5py
import numpy
import pandas
import sklearn.preprocessing


LabelEncoderPB = DatabasePB.LabelEncoder


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


def serialise_encoder(
        encoder: sklearn.preprocessing.LabelEncoder) -> LabelEncoderPB:
    """Serialises a LabelEncoder as a protobuf.

    Parameters
    ----------
    encoder
        LabelEncoder.

    Returns
    -------
    LabelEncoderPB
        Protobuf representing the LabelEncoder.
    """
    proto = LabelEncoderPB()
    if not hasattr(encoder, 'classes_'):
        return proto

    for i, class_label in enumerate(encoder.classes_):
        encoding = proto.encoding.add()
        encoding.class_label = str(class_label)
        encoding.class_int = i
    return proto


class Database(ABC):
    """Base class for database wrappers."""

    @abstractmethod
    def __enter__(self):
        return self

    @abstractmethod
    def __exit__(self, exc_type: Exception, exc_val: object, exc_tb: Traceback):
        pass

    @abstractmethod
    def read_features(self, ids: Sequence[int]) -> numpy.ndarray:
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
                    labeller_ids: Sequence[int],
                    instance_ids: Sequence[int]) -> numpy.ndarray:
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
    def write_features(self, ids: Sequence[int], features: numpy.ndarray):
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
                     labeller_ids: Sequence[int],
                     instance_ids: Sequence[int],
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
    def get_known_instance_ids(self) -> List[int]:
        """Returns a list of known instance IDs.

        Returns
        -------
        List[str]
            A list of known instance IDs.
        """

    @abstractmethod
    def get_known_labeller_ids(self) -> List[int]:
        """Returns a list of known labeller IDs.

        Returns
        -------
        List[str]
            A list of known labeller IDs.
        """

    @abstractmethod
    def to_proto(self) -> DatabasePB:
        """Serialises this database as a protobuf.

        Returns
        -------
        DatabasePB
            Protobuf representing this database.
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
    _h5_file : h5py.File
        Opened HDF5 file.
    _sync_attrs : List[str]
        List of instance attributes to sync with the HDF5 file's attributes.
    """

    def __init__(self, path: str, label_dtype: str=None,
                 feature_dtype: str=None):
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
        """
        super().__init__(path)
        self.label_dtype = label_dtype
        self._default_label_dtype = 'float32'
        self.feature_dtype = feature_dtype
        self._default_feature_dtype = 'float32'

        # List of attributes to keep in sync with the HDF5 file.
        self._sync_attrs = ['label_dtype', 'feature_dtype']

    def to_proto(self) -> DatabasePB:
        """Serialises this database as a protobuf.

        Returns
        -------
        DatabasePB
            Protobuf representing this database.
        """
        proto = DatabasePB()
        proto.path = self.path
        proto.class_name = 'ManagedHDF5Database'
        db_kwargs = {
            'label_dtype': self.label_dtype,
            'feature_dtype': self.feature_dtype}
        for key, value in db_kwargs.items():
            kwarg = proto.kwarg.add()
            kwarg.key = key
            kwarg.value = json.dumps(value)
        # No encoder for a managed DB - assume that labels are encoded already.
        # proto.label_encoder.CopyFrom(serialise_encoder(self.label_encoder))
        return proto

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

    def write_features(self, ids: Sequence[int], features: numpy.ndarray):
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

        # Early termination.
        if not ids:
            return

        # Cast the features to the right type.
        if features.dtype != self.feature_dtype:
            warnings.warn('Casting features from type {} to type {}.'.format(
                features.dtype, self.feature_dtype))
            features = features.astype(self.feature_dtype)

        # Resize the feature array if we need to store more IDs than before.
        max_id = max(ids) + 1
        if max_id > self._h5_file['features'].shape[0]:
            self._h5_file['features'].resize(
                (max_id, self._h5_file.attrs['n_features']))
        # Store the feature vectors.
        # TODO(MatthewJA): Vectorise this. This could be tricky as HDF5 doesn't
        # fully support NumPy's fancy indexing.
        for id_, feature in zip(ids, features):
            self._h5_file['features'][id_, :] = feature

        # Add the IDs to the database.
        known_ids = set(self.get_known_instance_ids())
        new_ids = [i for i in ids if i not in known_ids]
        n_new_ids = len(new_ids)
        n_old_ids = self._h5_file['instance_ids'].shape[0]
        self._h5_file['instance_ids'].resize((n_old_ids + n_new_ids,))
        self._h5_file['instance_ids'][-n_new_ids:] = numpy.array(
            new_ids, dtype=int)

    def read_features(self, ids: Sequence[int]) -> numpy.ndarray:
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
        features = self._h5_file['features'].value[ids, :]
        features = numpy.asarray(
            features, dtype=self._h5_file.attrs['feature_dtype'])
        return features

    def write_labels(self,
                     labeller_ids: Sequence[int],
                     instance_ids: Sequence[int],
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

        # Early termination.
        if not labeller_ids or not instance_ids:
            return

        # Cast the labels to the right type.
        if labels.dtype != self.label_dtype:
            warnings.warn('Casting labels from type {} to type {}.'.format(
                labels.dtype, self.label_dtype))
            labels = labels.astype(self.label_dtype)

        # Resize the label array if necessary.
        max_labeller_id = max(labeller_ids) + 1
        max_instance_id = max(instance_ids) + 1
        if (max_labeller_id > self._h5_file['labels'].shape[0] or
                max_instance_id > self._h5_file['labels'].shape[1]):
            self._h5_file['labels'].resize(
                (max_labeller_id, max_instance_id,
                 self._h5_file.attrs['label_dim']))
        # Store the labels.
        # TODO(MatthewJA): Vectorise this.
        for labeller_idx, labeller_id in enumerate(labeller_ids):
            for instance_idx, instance_id in enumerate(instance_ids):
                label = labels[labeller_idx, instance_idx]
                self._h5_file['labels'][
                    labeller_id, instance_id, :] = label

        logging.debug(
            'New label array size: {}'.format(self._h5_file['labels'].shape))

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
                new_instance_ids, dtype=int)

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
                new_labeller_ids, dtype=int)

    def read_labels(self,
                    labeller_ids: Sequence[int],
                    instance_ids: Sequence[int]) -> numpy.ndarray:
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

        labels = self._h5_file['labels'].value[labeller_ids][:, instance_ids, :]
        labels = numpy.asarray(labels, dtype=self._h5_file.attrs['label_dtype'])

        return labels

    def get_known_instance_ids(self) -> List[int]:
        """Returns a list of known instance IDs.

        Returns
        -------
        List[str]
            A list of known instance IDs.
        """
        self._assert_open()
        return [id_ for id_ in self._h5_file['instance_ids']]

    def get_known_labeller_ids(self) -> List[int]:
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
        if self.label_dtype is None:
            self.label_dtype = self._default_label_dtype
        if self.feature_dtype is None:
            self.feature_dtype = self._default_feature_dtype
        h5_file.create_dataset('features', shape=(0, 0),
                               dtype=self.feature_dtype,
                               maxshape=(None, None))
        h5_file.create_dataset('labels', shape=(0, 0, 0),
                               dtype=self.label_dtype,
                               maxshape=(None, None, None))
        h5_file.create_dataset('instance_ids', shape=(0,),
                               dtype=int, maxshape=(None,))
        h5_file.create_dataset('labeller_ids', shape=(0,),
                               dtype=int, maxshape=(None,))
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
            assert len(self._h5_file['features'].shape) == 2
            assert len(self._h5_file['labels'].shape) == 3
            assert len(self._h5_file['instance_ids'].shape) == 1
            assert len(self._h5_file['labeller_ids'].shape) == 1
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
    encode_labels : bool
        Whether to encode labels as integers.
    label_encoder : sklearn.preprocessing.LabelEncoder
        Encodes labels as integers.
    _h5_file : h5py.File
        HDF5 file object.
    _is_multidimensional : bool
        Whether the features are in a multidimensional dataset.
    """

    def __init__(self, path: str, feature_cols: List[str], label_col: str,
                 encode_labels: bool=True,
                 label_encoder: sklearn.preprocessing.LabelEncoder=None):
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
        encode_labels
            Whether to encode labels as integers.
        label_encoder
            Encodes labels as integers. If not specified, the label column will
            be read and a label encoding generated.
        """
        super().__init__(path)

        if not feature_cols:
            raise ValueError('Must specify feature columns for HDF5.')

        self.feature_cols = feature_cols
        self.label_col = label_col
        self.encode_labels = encode_labels
        self.label_encoder = label_encoder

        if self.label_encoder and not self.encode_labels:
            raise ValueError('label_encoder specified but encode_labels is '
                             'False')

        if self.label_encoder is None:
            self.label_encoder = sklearn.preprocessing.LabelEncoder()

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

    def to_proto(self) -> DatabasePB:
        """Serialises this database as a protobuf.

        Returns
        -------
        DatabasePB
            Protobuf representing this database.
        """
        proto = DatabasePB()
        proto.path = self.path
        proto.class_name = 'HDF5Reader'
        db_kwargs = {
            'feature_cols': self.feature_cols,
            'label_col': self.label_col,
            'encode_labels': self.encode_labels}
        for key, value in db_kwargs.items():
            kwarg = proto.kwarg.add()
            kwarg.key = key
            kwarg.value = json.dumps(value)
        proto.label_encoder.CopyFrom(serialise_encoder(self.label_encoder))
        return proto

    def read_features(self, ids: Sequence[int]) -> numpy.ndarray:
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
        # For each ID, get the corresponding features.
        if self._is_multidimensional:
            # If there are duplicates in ids, then this will crash with an
            # OSError! (and a very cryptic error message...) To get around this,
            # we'll first get all the unique IDs.
            unique_ids = []
            unique_ids_set = set()  # For lookups.
            id_to_index = {}  # For reconstructing the features.
            for id_ in ids:
                if id_ not in unique_ids_set:
                    unique_ids.append(id_)
                    unique_ids_set.add(id_)
                    id_to_index[id_] = len(unique_ids) - 1
            # Then index with just the unique IDs.
            features_ = self._h5_file[self.feature_cols[0]][unique_ids]
            # Finally, reconstruct the features array.
            features = numpy.zeros((len(ids), features_.shape[1]))
            for index, id_ in enumerate(ids):
                index_ = id_to_index[id_]
                features[index, :] = features_[index_, :]
            return features
        else:
            # Allocate output array.
            features = numpy.zeros((len(ids), len(self.feature_cols)))
            # Read each feature.
            features_h5 = self._h5_file[self.feature_cols[0]]
            for feature_idx, feature_name in enumerate(self.feature_cols):
                features[ids, feature_idx] = features_h5[feature_name][ids]
        return numpy.nan_to_num(features)

    def read_labels(self,
                    labeller_ids: Sequence[int],
                    instance_ids: Sequence[int]) -> numpy.ndarray:
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

        if len(labeller_ids) > 1:
            raise NotImplementedError('Multiple labellers not yet supported.')

        # TODO(MatthewJA): Optimise this.
        # For each ID, get the corresponding labels.
        # If there are duplicates in ids, then this will crash with an
        # OSError! (and a very cryptic error message...) To get around this,
        # we'll first get all the unique IDs.
        unique_ids = []
        unique_ids_set = set()  # For lookups.
        id_to_index = {}  # For reconstructing the labels.
        for id_ in instance_ids:
            if id_ not in unique_ids_set:
                unique_ids.append(id_)
                unique_ids_set.add(id_)
                id_to_index[id_] = len(unique_ids) - 1
        # Then index with just the unique IDs.
        labels_ = self._h5_file[self.label_col][unique_ids].reshape(
            (1, len(unique_ids), -1))
        # Finally, reconstruct the labels array.
        labels = numpy.zeros(
            (1, len(instance_ids), labels_.shape[2]),
            dtype=labels_.dtype)
        for index, id_ in enumerate(instance_ids):
            index_ = id_to_index[id_]
            labels[0, index, :] = labels_[0, index_, :]

        if labels.shape[2] != 1:
            raise NotImplementedError('Multidimensional labels not currently '
                                      'supported.')

        # Encode labels.
        if self.encode_labels:
            labels = numpy.apply_along_axis(
                self.label_encoder.fit_transform,
                axis=1,
                arr=labels.reshape(labels.shape[:2])
                ).reshape(labels.shape)

        return labels

    def write_features(self, ids: Sequence[int], features: numpy.ndarray):
        raise PermissionError('Cannot write to read-only database.')

    def write_labels(self,
                     labeller_ids: Sequence[int],
                     instance_ids: Sequence[int],
                     labels: numpy.ndarray):
        raise PermissionError('Cannot write to read-only database.')

    def get_known_instance_ids(self) -> List[int]:
        """Returns a list of known instance IDs.

        Returns
        -------
        List[str]
            A list of known instance IDs.
        """
        self._assert_open()
        return [i for i in range(self.n_instances)]

    def get_known_labeller_ids(self) -> List[int]:
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
    encode_labels : bool
        Whether to encode labels as integers.
    label_encoder : sklearn.preprocessing.LabelEncoder
        Encodes labels as integers.
    _db : Database
        Underlying ManagedHDF5Database.
    _db_filepath : str
        Path of underlying HDF5 database.
    _tempdir : str
        Temporary directory where the underlying HDF5 database is stored.
    """

    def __init__(self, path: str, feature_cols: List[str], label_col: str,
                 encode_labels: bool=True,
                 label_encoder: sklearn.preprocessing.LabelEncoder=None):
        """
        Parameters
        ----------
        path
            Path to ASCII file.
        feature_cols
            List of feature columns.
        label_col
            Name of label column.
        encode_labels
            Whether to encode labels as integers.
        label_encoder
            Encodes labels as integers. If not specified, the label column will
            be read and a label encoding generated.
        """
        self.path = path
        self.feature_cols = feature_cols
        self.label_col = label_col
        self.encode_labels = encode_labels
        self.label_encoder = label_encoder

        if self.label_encoder and not self.encode_labels:
            raise ValueError('label_encoder specified but encode_labels is '
                             'False')

        if self.label_encoder is None:
            self.label_encoder = sklearn.preprocessing.LabelEncoder()

    def to_proto(self) -> DatabasePB:
        """Serialises this database as a protobuf.

        Returns
        -------
        DatabasePB
            Protobuf representing this database.
        """
        proto = DatabasePB()
        proto.path = self.path
        proto.class_name = 'ASCIIReader'
        db_kwargs = {
            'feature_cols': self.feature_cols,
            'label_col': self.label_col,
            'encode_labels': self.encode_labels}
        for key, value in db_kwargs.items():
            kwarg = proto.kwarg.add()
            kwarg.key = key
            kwarg.value = json.dumps(value)
        proto.label_encoder.CopyFrom(serialise_encoder(self.label_encoder))
        return proto

    def _db_from_ascii(self,
                       db: Database,
                       data: astropy.table.Table,
                       feature_cols: Sequence[str],
                       label_col: str,
                       ids: Sequence[int]):
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
        ids
            List of instance IDs.
        """
        # Read in features.
        columns = data.keys()
        if not feature_cols:
            # If there are no features given, use all columns.
            feature_cols = [c for c in columns if c != label_col]

        # This converts the features from a table to an array.
        features = data[feature_cols].as_array()
        features = features.view(numpy.float64).reshape(features.shape + (-1,))

        # Read in labels.
        labels = numpy.array(
            data[label_col]).reshape((1, -1, 1))

        # We want to support multiple labellers in the future, but currently
        # don't. So every labeller is the same, ID = 0.
        labeller_ids = [0]

        # Encode labels.
        if self.encode_labels:
            labels = numpy.apply_along_axis(
                self.label_encoder.fit_transform,
                axis=1,
                arr=labels.reshape(labels.shape[:2])
                ).reshape(labels.shape)

        # Write to database.
        db.write_features(ids, features)
        db.write_labels(labeller_ids, ids, labels)

    def __enter__(self):
        self._tempdir = tempfile.TemporaryDirectory(prefix='acton')
        # Read the whole file into a DB.
        self._db_filepath = os.path.join(self._tempdir.name, 'db.h5')

        data = io_ascii.read(self.path)
        ids = list(range(len(data[self.label_col])))

        max_label_len = max(len(str(i)) for i in data[self.label_col])
        label_dtype = '<S{}'.format(max_label_len)

        self._db = ManagedHDF5Database(
            self._db_filepath,
            label_dtype=label_dtype,
            feature_dtype='float64')
        self._db.__enter__()
        try:
            # We want to handle the encoding ourselves.
            self._db_from_ascii(self._db, data, self.feature_cols,
                                self.label_col, ids, encode_labels=False)
        except TypeError:
            # Encoding isn't supported in the underlying database.
            self._db_from_ascii(self._db, data, self.feature_cols,
                                self.label_col, ids)
        return self

    def __exit__(self, exc_type: Exception, exc_val: object, exc_tb: Traceback):
        self._db.__exit__(exc_type, exc_val, exc_tb)
        self._tempdir.cleanup()
        delattr(self, '_db')

    def read_features(self, ids: Sequence[int]) -> numpy.ndarray:
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
                    labeller_ids: Sequence[int],
                    instance_ids: Sequence[int]) -> numpy.ndarray:
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
        # N.B. Labels are encoded in _db_from_ascii.
        return self._db.read_labels(labeller_ids, instance_ids)

    def write_features(self, ids: Sequence[int], features: numpy.ndarray):
        raise NotImplementedError('Cannot write to read-only database.')

    def write_labels(self,
                     labeller_ids: Sequence[int],
                     instance_ids: Sequence[int],
                     labels: numpy.ndarray):
        raise NotImplementedError('Cannot write to read-only database.')

    def get_known_instance_ids(self) -> List[int]:
        """Returns a list of known instance IDs.

        Returns
        -------
        List[str]
            A list of known instance IDs.
        """
        return self._db.get_known_instance_ids()

    def get_known_labeller_ids(self) -> List[int]:
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
    encode_labels : bool
        Whether to encode labels as integers.
    label_encoder : sklearn.preprocessing.LabelEncoder
        Encodes labels as integers.
    _df : pandas.DataFrame
        Pandas dataframe.
    """

    def __init__(self, path: str, feature_cols: List[str], label_col: str,
                 key: str, encode_labels: bool=True,
                 label_encoder: sklearn.preprocessing.LabelEncoder=None):
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
        encode_labels
            Whether to encode labels as integers.
        label_encoder
            Encodes labels as integers. If not specified, the label column will
            be read and a label encoding generated.
        """
        self.path = path
        self.feature_cols = feature_cols
        self.label_col = label_col
        self.key = key
        self._df = pandas.read_hdf(self.path, self.key)
        self.encode_labels = encode_labels
        self.label_encoder = label_encoder

        if self.label_encoder and not self.encode_labels:
            raise ValueError('label_encoder specified but encode_labels is '
                             'False')

        if self.label_encoder is None:
            self.label_encoder = sklearn.preprocessing.LabelEncoder()

        if not self.feature_cols:
            self.feature_cols = [k for k in self._df.keys()
                                 if k != self.label_col]

        self.n_instances = len(self._df[self.label_col])
        self.n_features = len(self.feature_cols)

    def to_proto(self) -> DatabasePB:
        """Serialises this database as a protobuf.

        Returns
        -------
        DatabasePB
            Protobuf representing this database.
        """
        proto = DatabasePB()
        proto.path = self.path
        proto.class_name = 'PandasReader'
        db_kwargs = {
            'feature_cols': self.feature_cols,
            'label_col': self.label_col,
            'key': self.key,
            'encode_labels': self.encode_labels}
        for key, value in db_kwargs.items():
            kwarg = proto.kwarg.add()
            kwarg.key = key
            kwarg.value = json.dumps(value)
        proto.label_encoder.CopyFrom(serialise_encoder(self.label_encoder))
        return proto

    def __enter__(self):
        return self

    def __exit__(self, exc_type: Exception, exc_val: object, exc_tb: Traceback):
        delattr(self, '_df')

    def read_features(self, ids: Sequence[int]) -> numpy.ndarray:
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
            sel = self._df.iloc[id_]

            for feature_index, feature in enumerate(self.feature_cols):
                features[out_index, feature_index] = sel[feature]

        return features

    def read_labels(self,
                    labeller_ids: Sequence[int],
                    instance_ids: Sequence[int]) -> numpy.ndarray:
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
        # Draw a label to get the dtype.
        dtype = type(self._df.iloc[0][self.label_col])

        # Allocate output labels array.
        labels = numpy.zeros(
            (len(labeller_ids), len(instance_ids), 1),
            dtype=dtype)

        if len(labeller_ids) > 1:
            raise NotImplementedError('Multiple labellers not yet supported.')

        # For each ID, get the corresponding labels.
        for out_index, id_ in enumerate(instance_ids):
            sel = self._df.iloc[int(id_)]
            labels[0, out_index, 0] = sel[self.label_col]

        if labels.shape[2] != 1:
            raise NotImplementedError('Multidimensional labels not currently '
                                      'supported.')

        # Encode labels.
        if self.encode_labels:
            labels = numpy.apply_along_axis(
                self.label_encoder.fit_transform,
                axis=1,
                arr=labels.reshape(labels.shape[:2])
                ).reshape(labels.shape)

        return labels

    def write_features(self, ids: Sequence[int], features: numpy.ndarray):
        raise PermissionError('Cannot write to read-only database.')

    def write_labels(self,
                     labeller_ids: Sequence[int],
                     instance_ids: Sequence[int],
                     labels: numpy.ndarray):
        raise PermissionError('Cannot write to read-only database.')

    def get_known_instance_ids(self) -> List[int]:
        """Returns a list of known instance IDs.

        Returns
        -------
        List[str]
            A list of known instance IDs.
        """
        return [i for i in range(self.n_instances)]

    def get_known_labeller_ids(self) -> List[int]:
        """Returns a list of known labeller IDs.

        Returns
        -------
        List[str]
            A list of known labeller IDs.
        """
        raise NotImplementedError()


class FITSReader(Database):
    """Reads FITS databases.

    Attributes
    ----------
    hdu_index : int
        Index of HDU in the FITS file.
    feature_cols : List[str]
        List of feature columns.
    label_col : str
        Name of label column.
    n_features : int
        Number of features.
    n_instances : int
        Number of instances.
    n_labels : int
        Number of labels per instance.
    path : str
        Path to FITS file.
    encode_labels : bool
        Whether to encode labels as integers.
    label_encoder : sklearn.preprocessing.LabelEncoder
        Encodes labels as integers.
    _hdulist : astropy.io.fits.HDUList
        FITS HDUList.
    """

    def __init__(self, path: str, feature_cols: List[str], label_col: str,
                 hdu_index: int=1, encode_labels: bool=True,
                 label_encoder: sklearn.preprocessing.LabelEncoder=None):
        """
        Parameters
        ----------
        path
            Path to FITS file.
        feature_cols
            List of feature columns. If none are specified, then all non-label,
            non-ID columns will be used.
        label_col
            Name of label dataset.
        hdu_index
            Index of HDU in the FITS file. Default is 1, i.e., the first
            extension in the FITS file.
        encode_labels
            Whether to encode labels as integers.
        label_encoder
            Encodes labels as integers. If not specified, the label column will
            be read and a label encoding generated.
        """
        self.path = path
        self.feature_cols = feature_cols
        self.label_col = label_col
        self.hdu_index = hdu_index
        self.encode_labels = encode_labels
        self.label_encoder = label_encoder

        if self.label_encoder and not self.encode_labels:
            raise ValueError('label_encoder specified but encode_labels is '
                             'False')

        if self.label_encoder is None:
            self.label_encoder = sklearn.preprocessing.LabelEncoder()

        # These will be set when the FITS file is opened.
        self.n_instances = None
        self.n_features = None

    def to_proto(self) -> DatabasePB:
        """Serialises this database as a protobuf.

        Returns
        -------
        DatabasePB
            Protobuf representing this database.
        """
        proto = DatabasePB()
        proto.path = self.path
        proto.class_name = 'FITSReader'
        db_kwargs = {
            'feature_cols': self.feature_cols,
            'label_col': self.label_col,
            'hdu_index': self.hdu_index,
            'encode_labels': self.encode_labels}
        for key, value in db_kwargs.items():
            kwarg = proto.kwarg.add()
            kwarg.key = key
            kwarg.value = json.dumps(value)
        proto.label_encoder.CopyFrom(serialise_encoder(self.label_encoder))
        return proto

    def __enter__(self):
        self._hdulist = io_fits.open(self.path)

        # If we haven't specified columns, use all except the label column.
        cols = self._hdulist[self.hdu_index].columns.names
        if not self.feature_cols:
            self.feature_cols = [k for k in cols if k != self.label_col]

        self.n_features = len(self.feature_cols)
        self.n_instances = \
            self._hdulist[self.hdu_index].data[self.label_col].ravel().shape[0]

        return self

    def __exit__(self, exc_type: Exception, exc_val: object, exc_tb: Traceback):
        self._hdulist.close()
        delattr(self, '_hdulist')

    def read_features(self, ids: Sequence[int]) -> numpy.ndarray:
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
        for f_index, col in enumerate(self.feature_cols):
            col = self._hdulist[self.hdu_index].data[col]
            features[:, f_index] = col[ids]

        return features

    def read_labels(self,
                    labeller_ids: Sequence[int],
                    instance_ids: Sequence[int]) -> numpy.ndarray:
        """Reads label vectors from the database.

        Parameters
        ----------
        labeller_ids
            Iterable of labeller IDs.
        instance_ids
            Iterable of instance IDs.

        Returns
        -------
        numpy.p
            T x N x 1 array of label vectors.
        """
        label_col = self._hdulist[self.hdu_index].data[self.label_col]
        labels = label_col[instance_ids].reshape((1, -1, 1))

        # Encode labels.
        if self.encode_labels:
            labels = numpy.apply_along_axis(
                self.label_encoder.fit_transform,
                axis=1,
                arr=labels.reshape(labels.shape[:2])
                ).reshape(labels.shape)

        return labels

    def write_features(self, ids: Sequence[int], features: numpy.ndarray):
        raise PermissionError('Cannot write to read-only database.')

    def write_labels(self,
                     labeller_ids: Sequence[int],
                     instance_ids: Sequence[int],
                     labels: numpy.ndarray):
        raise PermissionError('Cannot write to read-only database.')

    def get_known_instance_ids(self) -> List[int]:
        """Returns a list of known instance IDs.

        Returns
        -------
        List[str]
            A list of known instance IDs.
        """
        return [i for i in range(self.n_instances)]

    def get_known_labeller_ids(self) -> List[int]:
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
    'FITSReader': FITSReader,
    'ManagedHDF5Database': ManagedHDF5Database,
    'PandasReader': PandasReader,
}
