"""Predictor classes."""

import numpy.ndarray
import proto.predictors_pb2 as predictors_pb


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
