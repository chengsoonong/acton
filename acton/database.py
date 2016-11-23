"""Wrapper class for databases."""

from abc import ABC, abstractmethod
from inspect import Traceback
from typing import Iterable, List
import warnings

import h5py
import numpy


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
    """Database using an HDF5 file.

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
        self.path = path
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
        try:
            self._h5_file = h5py.File(self.path, 'r+')
        except OSError:
            with h5py.File(self.path, 'w') as h5_file:
                self._setup_hdf5(h5_file)
            self._h5_file = h5py.File(self.path, 'r+')

        # Load attrs from HDF5 file if we haven't specified them.
        for attr in self._sync_attrs:
            if getattr(self, attr) is None:
                setattr(self, attr, self._h5_file.attrs[attr])

        self._validate_hdf5()

    def __enter__(self):
        self._open_hdf5()
        return self

    def __exit__(self, exc_type: Exception, exc_val: object, exc_tb: Traceback):
        self._h5_file.close()
        delattr(self, '_h5_file')

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

    def _assert_open(self):
        """Asserts that the HDF5 file is ready to be read to/written from.

        Raises
        ------
        AssertionError
        """
        assert hasattr(self, '_h5_file'), ('HDF5 database must be used as a '
                                           'context manager.')

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


# For safe string-based access to database classes.
DATABASES = {
    'HDF5Database': HDF5Database,
}
