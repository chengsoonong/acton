"""Predictor classes."""

import acton.proto.predictors_pb2 as predictors_pb
from numpy import ndarray


class Predictor(object):
    """Base class for predictors.

    Attributes
    ----------
    """

    def __init__(self):
        raise NotImplementedError('Cannot instantiate base class: Predictor.')

    def fit(self, features: ndarray, labels: ndarray):
        """Fits the predictor to labelled data.

        Parameters
        ----------
        features
            An N x D array of feature vectors.
        labels
            An N x 1 array of corresponding labels.
        """
        pass

    def predict(self, features: ndarray) -> ndarray:
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
        ndarray
            An N x 1 array of corresponding predictions.
        """
        pass
