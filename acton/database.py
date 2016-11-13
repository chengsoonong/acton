"""Wrapper class for databases."""

from abc import ABC, abstractmethod
from inspect import Traceback
from typing import Iterable, List

import h5py
import numpy


class Database(ABC):
    """Base class for database wrappers."""

    @abstractmethod
    def read_features(self, ids: Iterable[str]) -> numpy.ndarray:
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
    def __enter__(self):
        return self

    @abstractmethod
    def __exit__(self, exc_type: Exception, exc_val: object, exc_tb: Traceback):
        pass

    @abstractmethod
    def write_features(self, ids: Iterable[str], features: numpy.ndarray):
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


class HDF5Database(Database):
    """Database using an HDF5 file.

    Attributes
    ----------
    path : str
        Path to HDF5 file.
    _h5_file : h5py.File
        Opened HDF5 file.
    """

    def __init__(self, path: str, label_dtype: str='float32',
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
        max_id_length
            Maximum length of the IDs this database will store. If not provided
            then it will be read from the database file; if the database file
            does not exist then a default value of 128 will be used.
        """
        self.path = path
        self.label_dtype = None
        self._default_label_dtype = 'float32'
        self.max_id_length = max_id_length
        self._default_max_id_length = 128

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
        if self.max_id_length is None:
            self.max_id_length = self._h5_file.attrs['max_id_length']
        if self.label_dtype is None:
            self.label_dtype = self._h5_file['labels'].dtype
        self._validate_hdf5()

    def __enter__(self):
        self._open_hdf5()
        return self

    def __exit__(self, exc_type: Exception, exc_val: object, exc_tb: Traceback):
        self._h5_file.close()
        delattr(self, '_h5_file')

    def write_features(self, ids: Iterable[str], features: numpy.ndarray):
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

        if len(ids) != len(features):
            raise ValueError('Must have same number of IDs and features.')

        n_instances, n_dimensions = self._h5_file['features'].shape

        # Check shape of features against the shape we can store in our
        # database. We can only store these features if they are either the same
        # shape as we already have, or if there are no features already stored.
        if ((n_instances, n_dimensions) != (0, 0) and
                n_dimensions != features.shape[1]):
            raise ValueError('Features must have {} dimensions.'.format(
                n_dimensions))

        if n_dimensions == 0:
            # Will resize later when adding new instances.
            n_dimensions = features.shape[1]

        # Separate out the IDs into vectors we need to update and vectors
        # we need to add.
        known_ids = self.get_known_ids()
        known_ids_set = set(known_ids)
        new_ids = set(ids)
        known_new_ids = known_ids_set & new_ids
        unknown_new_ids = new_ids - known_new_ids

        # Dictionary mapping IDs to the corresponding index.
        id_to_new_index = {id_: index for index, id_ in enumerate(ids)}

        # Update the vectors of known IDs.
        updates = []
        for to_index, id_ in enumerate(self._h5_file['ids'].value):
            if id_ in known_new_ids:
                from_index = id_to_new_index[id_]
                updates.append((to_index, features[from_index]))
        for index, feature_vector in updates:
            self._h5_file['features'][index] = feature_vector

        # Resize the HDF5 file to fit the new features.
        n_new_instances = len(unknown_new_ids) + n_instances
        self._h5_file['features'].resize((n_new_instances, n_dimensions))
        self._h5_file['labels'].resize(
            (n_new_instances,
             self._h5_file['labels'].shape[1],
             self._h5_file['labels'].shape[2]))
        self._h5_file['ids'].resize((n_new_instances,))

        # Add on the new features.
        new_ids = sorted(unknown_new_ids, key=id_to_new_index.get)
        from_indices = [id_to_new_index[id_] for id_ in new_ids]
        new_features = features[from_indices]
        new_ids = numpy.array(new_ids, dtype=self._h5_file['ids'].dtype)

        # If there are no new features (only updates), we're done.
        if len(unknown_new_ids) == 0:
            return

        self._h5_file['features'][-len(unknown_new_ids):, :] = new_features
        self._h5_file['ids'][-len(unknown_new_ids):] = new_ids

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
        id_to_index = {id_: index for index, id_ in enumerate(ids)}
        # Indices of the features associated with the given IDs.
        indices = []
        ids_set = set(ids)
        # IDs sorted by their order in the HDF5 file.
        h5_sorted_ids = []
        for index, id_ in enumerate(self._h5_file['ids']):
            if id_ in ids_set:
                indices.append(index)
                h5_sorted_ids.append(id_)
        features = self._h5_file['features'][indices, :]
        # Now we need to sort these features by the given order of IDs.
        # Note that h5_sorted_ids are in the same order as features.
        # We want to find the ordering that turns h5_sorted_ids into ids, then
        # apply this ordering to features.
        h5_sorted_tuples = [(id_, idx) for idx, id_ in enumerate(h5_sorted_ids)]
        sorted_tuples = sorted(
            h5_sorted_tuples, key=lambda z: id_to_index[z[0]])
        # The second element of these tuples is now the correct order of
        # indices.
        sorted_indices = [i[1] for i in sorted_tuples]
        return features[sorted_indices]

    def get_known_ids(self) -> List[bytes]:
        """Returns a list of known IDs.

        Returns
        -------
        List[str]
            A list of known IDs.
        """
        self._assert_open()
        return [id_ for id_ in self._h5_file['ids']]

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
        h5_file.create_dataset('features', shape=(0, 0),
                               dtype='float32',
                               maxshape=(None, None))
        h5_file.create_dataset('labels', shape=(0, 0, 0),
                               dtype=self.label_dtype,
                               maxshape=(None, None, None))
        h5_file.create_dataset('ids', shape=(0,),
                               dtype='<S{}'.format(self.max_id_length),
                               maxshape=(None,))
        h5_file.attrs['max_id_length'] = self.max_id_length

    def _validate_hdf5(self):
        """Checks that self._h5_file has the correct schema.

        Raises
        ------
        ValueError
        """
        try:
            assert 'features' in self._h5_file
            assert 'labels' in self._h5_file
            assert 'ids' in self._h5_file
        except AssertionError:
            raise ValueError(
                'File {} is not a valid database.'.format(self.path))

        assert self.max_id_length is not None

        try:
            assert self._h5_file['ids'].dtype == \
                '<S{}'.format(self.max_id_length)
        except AssertionError:
            raise ValueError('Database {} has incompatible maximum ID length.')

        try:
            assert self._h5_file['labels'].dtype == self.label_dtype
        except AssertionError:
            raise ValueError('Database {} has incompatible label type: '.format(
                self._h5_file['labels'].dtype))
