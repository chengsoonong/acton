"""Predictor classes."""

from abc import ABC, abstractmethod
from typing import Union

import acton.database
import acton.proto.predictors_pb2 as predictors_pb
import acton.proto.io
import numpy
import sklearn.linear_model


class PredictorInput(object):
    """Wrapper for the predictor input protobuf.

    Attributes
    ----------
    proto : predictors_pb.Labels
        Protobuf representing the input.
    """

    def __init__(self, proto: Union[str, predictors_pb.Labels]):
        """
        Parameters
        ----------
        proto
            Path to .proto file, or raw protobuf itself.
        """
        try:
            self.proto = acton.proto.io.read_proto(proto, predictors_pb.Labels)
        except TypeError:
            if isinstance(proto, predictors_pb.Labels):
                self.proto = proto
            else:
                raise TypeError('proto should be str or protobuf.')
        self._validate_proto()

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

        self._labels = []
        for instance in self.proto.instance:
            data = instance.label
            shape = (self.proto.n_labellers, self.proto.n_label_dimensions)
            dtype = self.proto.dtype
            self._labels.append(acton.proto.io.get_ndarray(data, shape, dtype))
        self._labels = numpy.array(self._labels).transpose((1, 0, 2))
        return self._labels

    @property
    def features(self) -> numpy.ndarray:
        """Gets features array specified in input.

        Notes
        -----
        The returned array is cached by this object so future calls will not
        need to recompile the array.

        Returns
        -------
        numpy.ndarray
            N x D NumPy array of features.
        """
        if hasattr(self, '_features'):
            return self._features

        ids = [instance.id for instance in self.proto.instance]
        DB = acton.database.DATABASES[self.proto.db_class]
        with DB(self.proto.db_path, label_dtype=self.proto.dtype) as db:
            return db.read_features(ids)

    def _validate_proto(self):
        """Checks that the protobuf is valid and enforces constraints.

        Raises
        ------
        ValueError
        """
        if self.proto.db_class not in acton.database.DATABASES:
            raise ValueError('Invalid database class: {}'.format(
                self.proto.db_class))

        if self.proto.n_labellers < 1:
            raise ValueError('Number of labellers must be > 0.')

        if self.proto.n_label_dimensions < 1:
            raise ValueError('Label dimension must be > 0.')

        if not self.proto.db_path:
            raise ValueError('Must specify db_path.')

        if self.proto.n_label_dimensions > 1:
            raise NotImplementedError(
                'Multidimensional labels are not currently supported.')

    def _set_default(self):
        """Adds default parameters to the protobuf."""
        self.proto.dtype = self.proto.dtype or 'float32'


class Predictor(ABC):
    """Base class for predictors.

    Attributes
    ----------
    """

    @abstractmethod
    def fit(self, features: numpy.ndarray, labels: numpy.ndarray):
        """Fits the predictor to labelled data.

        Parameters
        ----------
        features
            An N x D array of feature vectors.
        labels
            An N x 1 array of corresponding labels.
        """

    def fit_proto(
            self,
            proto: Union[str, predictors_pb.Labels, PredictorInput]):
        """Fits the predictor to labelled data given by a protobuf.

        Parameters
        ----------
        proto
            Protobuf of labelled instances.
        """
        try:
            proto = PredictorInput(proto)
        except TypeError:
            pass

        return self.fit(proto.features, proto.labels)

    @abstractmethod
    def predict(self, features: numpy.ndarray) -> numpy.ndarray:
        """Predicts labels of instances.

        Notes
        -----
            Unlike in scikit-learn, predictions are always real-valued.
            Predicted labels for a classification problem are represented by
            predicted probabilities of the positive class.

        Parameters
        ----------
        features
            An N x D array of feature vectors.

        Returns
        -------
        numpy.ndarray
            An N x 1 array of corresponding predictions.
        """


class LogisticRegression(Predictor):
    """Logistic regression predictor.

    Notes
    -----
    This predictor wraps sklearn.linear_model.LogisticRegression.

    Attributes
    ----------
    _lr : sklearn.linear_model.LogisticRegression
        Underlying logistic regression model.
    """

    def __init__(self, **kwargs: dict):
        """
        Parameters
        ----------
        kwargs
            Keyword arguments passed to the underlying
            sklearn.linear_model.LogisticRegression object.
        """
        self._lr = sklearn.linear_model.LogisticRegression(**kwargs)

    def fit(self, features: numpy.ndarray, labels: numpy.ndarray):
        """Fits the predictor to labelled data.

        Parameters
        ----------
        features
            An N x D array of feature vectors.
        labels
            An N x 1 array of corresponding labels.
        """
        assert labels.shape[1] == 1 and len(labels.shape) == 2
        self._lr.fit(features, labels.ravel())

    def predict(self, features: numpy.ndarray) -> numpy.ndarray:
        """Predicts labels of instances.

        Notes
        -----
            Unlike in scikit-learn, predictions are always real-valued.
            Predicted labels for the classification problem are represented by
            predicted probabilities of the positive class.

        Parameters
        ----------
        features
            An N x D array of feature vectors.

        Returns
        -------
        numpy.ndarray
            An N x 1 array of corresponding predictions.
        """
        return self._lr.predict_proba(features)[:, 1:]


class LogisticRegressionCommittee(Predictor):
    """Logistic regression committee-based predictor.

    Notes
    -----
    This predictor wraps sklearn.linear_model.LogisticRegression.

    Attributes
    ----------
    n_classifiers : int
        Number of logistic regression classifiers in the committee.
    _committee : List[sklearn.linear_model.LogisticRegression]
        Underlying committee of logistic regression classifiers.
    """

    def __init__(self, n_classifiers: int=10, **kwargs: dict):
        """
        Parameters
        ----------
        n_classifiers
            Number of logistic regression classifiers in the committee.
        kwargs
            Keyword arguments passed to the underlying
            sklearn.linear_model.LogisticRegression object.
        """
        self._committee = [sklearn.linear_model.LogisticRegression(**kwargs)
                           for _ in range(n_classifiers)]
        self.n_classifiers = n_classifiers

    def fit(self, features: numpy.ndarray, labels: numpy.ndarray):
        """Fits the predictor to labelled data.

        Parameters
        ----------
        features
            An N x D array of feature vectors.
        labels
            An N x 1 array of corresponding labels.
        """
        # TODO(MatthewJA): Introduce committee variety.
        assert labels.shape[1] == 1 and len(labels.shape) == 2
        for classifier in self._committee:
            classifier.fit(features, labels.ravel())

    def predict(self, features: numpy.ndarray) -> numpy.ndarray:
        """Predicts labels of instances.

        Notes
        -----
            Unlike in scikit-learn, predictions are always real-valued.
            Predicted labels for the classification problem are represented by
            predicted probabilities of the positive class.

        Parameters
        ----------
        features
            An N x D array of feature vectors.

        Returns
        -------
        numpy.ndarray
            An N x T array of corresponding predictions.
        """
        predictions = numpy.concatenate(
            [classifier.predict_proba(features)[:, 1:]
             for classifier in self._committee],
            axis=1)
        assert predictions.shape == (features.shape[0], self.n_classifiers)
        return predictions
