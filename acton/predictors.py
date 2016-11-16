"""Predictor classes."""

from abc import ABC, abstractmethod

import numpy
import sklearn.linear_model


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
