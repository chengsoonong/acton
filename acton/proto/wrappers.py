"""Classes that wrap protobufs."""

import json
from typing import Union, List, Iterable

import acton.database
import acton.proto.acton_pb2 as acton_pb
import acton.proto.io
import google.protobuf.json_format as json_format
import numpy


def validate_db(db: acton_pb.Database):
    """Validates a Database proto.

    Parameters
    ----------
    db
        Database to validate.

    Raises
    ------
    ValueError
    """
    if db.class_name not in acton.database.DATABASES:
        raise ValueError('Invalid database class name: {}'.format(
            db.class_name))

    if not db.path:
        raise ValueError('Must specify db.path.')


class LabelPool(object):
    """Wrapper for the LabelPool protobuf.

    Attributes
    ----------
    proto : acton_pb.LabelPool
        Protobuf representing the label pool.
    db_kwargs : dict
        Key-value pairs of keyword arguments for the database constructor.
    """

    def __init__(self, proto: Union[str, acton_pb.LabelPool]):
        """
        Parameters
        ----------
        proto
            Path to .proto file, or raw protobuf itself.
        """
        try:
            self.proto = acton.proto.io.read_proto(proto, acton_pb.LabelPool)
        except TypeError:
            if isinstance(proto, acton_pb.LabelPool):
                self.proto = proto
            else:
                raise TypeError('proto should be str or LabelPool protobuf.')
        self._validate_proto()
        self.db_kwargs = {kwa.key: json.loads(kwa.value)
                          for kwa in self.proto.db.kwarg}
        self._set_default()

    @classmethod
    def deserialise(cls, proto: bytes, json: bool=False) -> 'LabelPool':
        """Deserialises a protobuf into a LabelPool.

        Parameters
        ----------
        proto
            Serialised protobuf.
        json
            Whether the serialised protobuf is in JSON format.

        Returns
        -------
        LabelPool
        """
        if not json:
            lp = acton_pb.LabelPool()
            lp.ParseFromString(proto)
            return cls(lp)

        return cls(json_format.Parse(proto, acton_pb.LabelPool()))

    @property
    def DB(self) -> acton.database.Database:
        """Gets a database context manager for the specified database.

        Returns
        -------
        type
            Database context manager.
        """
        if hasattr(self, '_DB'):
            return self._DB

        self._DB = lambda: acton.database.DATABASES[self.proto.db.class_name](
            self.proto.db.path, **self.db_kwargs)

        return self._DB

    @property
    def ids(self) -> List[int]:
        """Gets a list of IDs.

        Returns
        -------
        List[int]
            List of known IDs.
        """
        if hasattr(self, '_ids'):
            return self._ids

        self._ids = list(self.proto.id)
        return self._ids

    @property
    def labels(self) -> numpy.ndarray:
        """Gets labels array specified in input.

        Notes
        -----
        The returned array is cached by this object so future calls will not
        need to recompile the array.

        Returns
        -------
        numpy.ndarray
            T x N x F NumPy array of labels.
        """
        if hasattr(self, '_labels'):
            return self._labels

        ids = self.ids
        with self.DB() as db:
            return db.read_labels([0], ids)

    def _validate_proto(self):
        """Checks that the protobuf is valid and enforces constraints.

        Raises
        ------
        ValueError
        """
        validate_db(self.proto.db)

    def _set_default(self):
        """Adds default parameters to the protobuf."""
        pass

    @classmethod
    def make(
            cls: type,
            ids: Iterable[int],
            db_path: str='',
            db_class: str='',
            db_kwargs: dict=None) -> 'LabelPool':
        """Constructs a LabelPool.

        Parameters
        ----------
        ids
            Iterable of instance IDs.
        db_path
            Path to database file.
        db_class
            Name of database class.
        db_kwargs
            Keyword arguments for the database constructor. Values must be
            JSON-stringifiable.

        Returns
        -------
        LabelPool
        """
        proto = acton_pb.LabelPool()

        # Handle default mutable arguments.
        db_kwargs = db_kwargs or {}

        # Store single data first.
        proto.db.path = db_path
        proto.db.class_name = db_class

        # Store the IDs.
        for id_ in ids:
            proto.id.append(id_)

        # Store the db_kwargs.
        for key, value in db_kwargs.items():
            kwarg = proto.db.kwarg.add()
            kwarg.key = key
            kwarg.value = json.dumps(value)

        return cls(proto)


class Predictions(object):
    """Wrapper for the Predictions protobuf.

    Attributes
    ----------
    proto : acton_pb.Predictions
        Protobuf representing predictions.
    """

    def __init__(self, proto: Union[str, acton_pb.Predictions]):
        """
        Parameters
        ----------
        proto
            Path to .proto file, or raw protobuf itself.
        """
        try:
            self.proto = acton.proto.io.read_proto(
                proto, acton_pb.Predictions)
        except TypeError:
            if isinstance(proto, acton_pb.Predictions):
                self.proto = proto
            else:
                raise TypeError('proto should be str or Predictions protobuf.')
        self._validate_proto()
        self.db_kwargs = {kwa.key: json.loads(kwa.value)
                          for kwa in self.proto.db.kwarg}
        self._set_default()

    @property
    def DB(self) -> acton.database.Database:
        """Gets a database context manager for the specified database.

        Returns
        -------
        type
            Database context manager.
        """
        if hasattr(self, '_DB'):
            return self._DB

        self._DB = lambda: acton.database.DATABASES[self.proto.db.class_name](
            self.proto.db.path, **self.db_kwargs)

        return self._DB

    @property
    def predicted_ids(self) -> List[int]:
        """Gets a list of IDs corresponding to predictions.

        Returns
        -------
        List[int]
            List of IDs corresponding to predictions.
        """
        if hasattr(self, '_predicted_ids'):
            return self._predicted_ids

        self._predicted_ids = [prediction.id
                               for prediction in self.proto.prediction]
        return self._predicted_ids

    @property
    def labelled_ids(self) -> List[int]:
        """Gets a list of IDs the predictor knew the label for.

        Returns
        -------
        List[int]
            List of IDs the predictor knew the label for.
        """
        if hasattr(self, '_labelled_ids'):
            return self._labelled_ids

        self._labelled_ids = list(self.proto.labelled_id)
        return self._labelled_ids

    @property
    def predictions(self) -> numpy.ndarray:
        """Gets predictions array specified in input.

        Notes
        -----
        The returned array is cached by this object so future calls will not
        need to recompile the array.

        Returns
        -------
        numpy.ndarray
            T x N x D NumPy array of predictions.
        """
        if hasattr(self, '_predictions'):
            return self._predictions

        self._predictions = []
        for prediction in self.proto.prediction:
            data = prediction.prediction
            shape = (self.proto.n_predictors,
                     self.proto.n_prediction_dimensions)
            self._predictions.append(
                acton.proto.io.get_ndarray(data, shape, float))
        self._predictions = numpy.array(self._predictions).transpose((1, 0, 2))
        return self._predictions

    def _validate_proto(self):
        """Checks that the protobuf is valid and enforces constraints.

        Raises
        ------
        ValueError
        """
        if self.proto.n_predictors < 1:
            raise ValueError('Number of predictors must be > 0.')

        if self.proto.n_prediction_dimensions < 1:
            raise ValueError('Prediction dimension must be > 0.')

        validate_db(self.proto.db)

    def _set_default(self):
        """Adds default parameters to the protobuf."""
        pass

    @classmethod
    def make(
            cls: type,
            predicted_ids: Iterable[int],
            labelled_ids: Iterable[int],
            predictions: numpy.ndarray,
            predictor: str='',
            db_path: str='',
            db_class: str='',
            db_kwargs: dict=None) -> 'Predictions':
        """Converts NumPy predictions to a Predictions object.

        Parameters
        ----------
        predicted_ids
            Iterable of instance IDs corresponding to predictions.
        labelled_ids
            Iterable of instance IDs used to train the predictor.
        predictions
            T x N x D array of corresponding predictions.
        predictor
            Name of predictor used to generate predictions.
        db_path
            Path to database file.
        db_class
            Name of database class.
        db_kwargs
            Keyword arguments for the database constructor. Values must be
            JSON-stringifiable.

        Returns
        -------
        Predictions
        """
        proto = acton_pb.Predictions()

        # Handle default mutable arguments.
        db_kwargs = db_kwargs or {}

        # Store single data first.
        n_predictors, n_instances, n_prediction_dimensions = predictions.shape
        proto.n_predictors = n_predictors
        proto.n_prediction_dimensions = n_prediction_dimensions
        proto.predictor = predictor
        proto.db.path = db_path
        proto.db.class_name = db_class

        # Store the predictions array. We can do this by looping over the
        # instances.
        for id_, prediction in zip(
                predicted_ids, predictions.transpose((1, 0, 2))):
            prediction_ = proto.prediction.add()
            prediction_.id = int(id_)  # numpy.int64 -> int
            prediction_.prediction.extend(prediction.ravel())

        # Store the labelled IDs.
        for id_ in labelled_ids:
            # int() here takes numpy.int64 to int, for protobuf compatibility.
            proto.labelled_id.append(int(id_))

        # Store the db_kwargs.
        for key, value in db_kwargs.items():
            kwarg = proto.db.kwarg.add()
            kwarg.key = key
            kwarg.value = json.dumps(value)

        return cls(proto)

    @classmethod
    def deserialise(cls, proto: bytes, json: bool=False) -> 'Predictions':
        """Deserialises a protobuf into Predictions.

        Parameters
        ----------
        proto
            Serialised protobuf.
        json
            Whether the serialised protobuf is in JSON format.

        Returns
        -------
        Predictions
        """
        if not json:
            predictions = acton_pb.Predictions()
            predictions.ParseFromString(proto)
            return cls(predictions)

        return cls(json_format.Parse(proto, acton_pb.Predictions()))


class Recommendations(object):
    """Wrapper for the Recommendations protobuf.

    Attributes
    ----------
    proto : acton_pb.Recommendations
        Protobuf representing recommendations.
    db_kwargs : dict
        Key-value pairs of keyword arguments for the database constructor.
    """

    def __init__(self, proto: Union[str, acton_pb.Recommendations]):
        """
        Parameters
        ----------
        proto
            Path to .proto file, or raw protobuf itself.
        """
        try:
            self.proto = acton.proto.io.read_proto(
                proto, acton_pb.Recommendations)
        except TypeError:
            if isinstance(proto, acton_pb.Recommendations):
                self.proto = proto
            else:
                raise TypeError(
                    'proto should be str or Recommendations protobuf.')
        self._validate_proto()
        self.db_kwargs = {kwa.key: json.loads(kwa.value)
                          for kwa in self.proto.db.kwarg}
        self._set_default()

    @classmethod
    def deserialise(cls, proto: bytes, json: bool=False) -> 'Recommendations':
        """Deserialises a protobuf into Recommendations.

        Parameters
        ----------
        proto
            Serialised protobuf.
        json
            Whether the serialised protobuf is in JSON format.

        Returns
        -------
        Recommendations
        """
        if not json:
            recommendations = acton_pb.Recommendations()
            recommendations.ParseFromString(proto)
            return cls(recommendations)

        return cls(json_format.Parse(proto, acton_pb.Recommendations()))

    @property
    def DB(self) -> acton.database.Database:
        """Gets a database context manager for the specified database.

        Returns
        -------
        type
            Database context manager.
        """
        if hasattr(self, '_DB'):
            return self._DB

        self._DB = lambda: acton.database.DATABASES[self.proto.db.class_name](
            self.proto.db.path, **self.db_kwargs)

        return self._DB

    @property
    def recommendations(self) -> List[int]:
        """Gets a list of recommended IDs.

        Returns
        -------
        List[int]
            List of recommended IDs.
        """
        if hasattr(self, '_recommendations'):
            return self._recommendations

        self._recommendations = list(self.proto.recommended_id)
        return self._recommendations

    @property
    def labelled_ids(self) -> List[int]:
        """Gets a list of labelled IDs.

        Returns
        -------
        List[int]
            List of labelled IDs.
        """
        if hasattr(self, '_labelled_ids'):
            return self._labelled_ids

        self._labelled_ids = list(self.proto.labelled_id)
        return self._labelled_ids

    def _validate_proto(self):
        """Checks that the protobuf is valid and enforces constraints.

        Raises
        ------
        ValueError
        """
        validate_db(self.proto.db)

    def _set_default(self):
        """Adds default parameters to the protobuf."""
        pass

    @classmethod
    def make(
            cls: type,
            recommended_ids: Iterable[int],
            labelled_ids: Iterable[int],
            recommender: str,
            db_path: str='',
            db_class: str='',
            db_kwargs: dict=None) -> 'Recommendations':
        """Constructs a LabelPool.

        Parameters
        ----------
        recommended_ids
            Iterable of recommended instance IDs.
        labelled_ids
            Iterable of labelled instance IDs used to make recommendations.
        recommender
            Name of the recommender used to make recommendations.
        db_path
            Path to database file.
        db_class
            Name of database class.
        db_kwargs
            Keyword arguments for the database constructor. Values must be
            JSON-stringifiable.

        Returns
        -------
        Recommendations
        """
        proto = acton_pb.Recommendations()

        # Handle default mutable arguments.
        db_kwargs = db_kwargs or {}

        # Store single data first.
        proto.recommender = recommender
        proto.db.path = db_path
        proto.db.class_name = db_class

        # Store the IDs.
        for id_ in recommended_ids:
            proto.recommended_id.append(id_)
        for id_ in labelled_ids:
            proto.labelled_id.append(id_)

        # Store the db_kwargs.
        for key, value in db_kwargs.items():
            kwarg = proto.db.kwarg.add()
            kwarg.key = key
            kwarg.value = json.dumps(value)

        return cls(proto)
