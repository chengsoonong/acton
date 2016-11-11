"""Predictor classes."""

import acton.proto.predictors_pb2 as predictors_pb
import acton.proto.io
import numpy


class PredictorInput(object):
    """Wrapper for the predictor input protobuf.

    Attributes
    ----------
    proto : predictors_pb.Labels
        Protobuf representing the input.
    """

    def __init__(self, path: str):
        """
        Parameters
        ----------
        path
            Path to .proto file.
        """
        self.proto = acton.proto.io.read_proto(path, predictors_pb.Labels)

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

        # TODO(MatthewJA): Connect a database.
        raise NotImplementedError()

        # self._features = []
        # for instance in self.proto.instance:
        #     # acton.proto.io.get_ndarray(self.proto.)


class Predictor(object):
    """Base class for predictors.

    Attributes
    ----------
    """

    def __init__(self):
        raise NotImplementedError('Cannot instantiate base class: Predictor.')

    def fit(self, features: numpy.ndarray, labels: numpy.ndarray):
        """Fits the predictor to labelled data.

        Parameters
        ----------
        features
            An N x D array of feature vectors.
        labels
            An N x 1 array of corresponding labels.
        """
        pass

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
        pass
