"""Classes that wrap protobufs."""

import json
from typing import Union, List, Iterable

import acton.database
import acton.proto.acton_pb2 as acton_pb
import acton.proto.io
import google.protobuf.json_format as json_format
import numpy
import sklearn.preprocessing
from sklearn.preprocessing import LabelEncoder as SKLabelEncoder


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


def deserialise_encoder(
            encoder: acton_pb.Database.LabelEncoder
        ) -> sklearn.preprocessing.LabelEncoder:
    """Deserialises a LabelEncoder protobuf.

    Parameters
    ----------
    encoder
        LabelEncoder protobuf.

    Returns
    -------
    sklearn.preprocessing.LabelEncoder
        LabelEncoder (or None if no encodings were specified).
    """
    encodings = []
    for encoding in encoder.encoding:
        encodings.append((encoding.class_int, encoding.class_label))
    encodings.sort()
    encodings = numpy.array([c[1] for c in encodings])

    encoder = SKLabelEncoder()
    encoder.classes_ = encodings
    return encoder


class LabelPool(object):
    """Wrapper for the LabelPool protobuf.

    Attributes
    ----------
    proto : acton_pb.LabelPool
        Protobuf representing the label pool.
    db_kwargs : dict
        Key-value pairs of keyword arguments for the database constructor.
    label_encoder : sklearn.preprocessing.LabelEncoder
        Encodes labels as integers. May be None.
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
        if len(self.proto.db.label_encoder.encoding) > 0:
            self.label_encoder = deserialise_encoder(
                self.proto.db.label_encoder)
            self.db_kwargs['label_encoder'] = self.label_encoder
        else:
            self.label_encoder = None
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
            db: acton.database.Database) -> 'LabelPool':
        """Constructs a LabelPool.

        Parameters
        ----------
        ids
            Iterable of instance IDs.
        db
            Database

        Returns
        -------
        LabelPool
        """
        proto = acton_pb.LabelPool()

        # Store the IDs.
        for id_ in ids:
            proto.id.append(id_)

        # Store the database.
        proto.db.CopyFrom(db.to_proto())

        return cls(proto)


class Predictions(object):
    """Wrapper for the Predictions protobuf.

    Attributes
    ----------
    proto : acton_pb.Predictions
        Protobuf representing predictions.
    db_kwargs : dict
        Dictionary of database keyword arguments.
    label_encoder : sklearn.preprocessing.LabelEncoder
        Encodes labels as integers. May be None.
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
        if len(self.proto.db.label_encoder.encoding) > 0:
            self.label_encoder = deserialise_encoder(
                self.proto.db.label_encoder)
            self.db_kwargs['label_encoder'] = self.label_encoder
        else:
            self.label_encoder = None
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
            db: acton.database.Database,
            predictor: str='') -> 'Predictions':
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
        db
            Database.

        Returns
        -------
        Predictions
        """
        proto = acton_pb.Predictions()

        # Store single data first.
        n_predictors, n_instances, n_prediction_dimensions = predictions.shape
        proto.n_predictors = n_predictors
        proto.n_prediction_dimensions = n_prediction_dimensions
        proto.predictor = predictor

        # Store the database.
        proto.db.CopyFrom(db.to_proto())

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
    label_encoder : sklearn.preprocessing.LabelEncoder
        Encodes labels as integers. May be None.
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
        if len(self.proto.db.label_encoder.encoding) > 0:
            self.label_encoder = deserialise_encoder(
                self.proto.db.label_encoder)
            self.db_kwargs['label_encoder'] = self.label_encoder
        else:
            self.label_encoder = None
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
            db: acton.database.Database) -> 'Recommendations':
        """Constructs a Recommendations.

        Parameters
        ----------
        recommended_ids
            Iterable of recommended instance IDs.
        labelled_ids
            Iterable of labelled instance IDs used to make recommendations.
        recommender
            Name of the recommender used to make recommendations.
        db
            Database.

        Returns
        -------
        Recommendations
        """
        proto = acton_pb.Recommendations()

        # Store single data first.
        proto.recommender = recommender

        # Store the IDs.
        for id_ in recommended_ids:
            proto.recommended_id.append(id_)
        for id_ in labelled_ids:
            proto.labelled_id.append(id_)

        # Store the database.
        proto.db.CopyFrom(db.to_proto())

        return cls(proto)
