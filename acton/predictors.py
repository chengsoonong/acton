"""Predictor classes."""

from abc import ABC, abstractmethod

import numpy
import sklearn.base


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
            An N x T array of corresponding predictions.
        """


class _InstancePredictor(Predictor):
    """Wrapper for a scikit-learn instance.

    Attributes
    ----------
    _instance : sklearn.base.BaseEstimator
        scikit-learn predictor instance.
    """

    def __init__(self, instance: sklearn.base.BaseEstimator):
        """
        Arguments
        ---------
        instance
            scikit-learn predictor instance.
        """
        self._instance = instance

    def fit(self, features: numpy.ndarray, labels: numpy.ndarray):
        """Fits the predictor to labelled data.

        Parameters
        ----------
        features
            An N x D array of feature vectors.
        labels
            An N x 1 array of corresponding labels.
        """
        self._instance.fit(features, labels.ravel())

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
            An N x T array of corresponding predictions.
        """
        return self._instance.predict_proba(features)[:, 1:]


def from_instance(predictor: sklearn.base.BaseEstimator) -> Predictor:
    """Converts a scikit-learn predictor instance into a Predictor instance.

    Arguments
    ---------
    predictor
        scikit-learn predictor.

    Returns
    -------
    Predictor
        Predictor instance wrapping the scikit-learn predictor.
    """
    return _InstancePredictor(predictor)


def from_class(Predictor: type) -> type:
    """Converts a scikit-learn predictor class into a Predictor class.

    Arguments
    ---------
    Predictor
        scikit-learn predictor class.

    Returns
    -------
    type
        Predictor class wrapping the scikit-learn class.
    """
    class Predictor_(_InstancePredictor):

        def __init__(self, **kwargs):
            super().__init__(instance=None)
            self._instance = Predictor(**kwargs)

    return Predictor_


class Committee(Predictor):
    """A predictor using a committee of other predictors.

    Attributes
    ----------
    n_classifiers : int
        Number of logistic regression classifiers in the committee.
    _committee : List[sklearn.linear_model.LogisticRegression]
        Underlying committee of logistic regression classifiers.
    """

    def __init__(self, Predictor: type, n_classifiers: int=10,
                 **kwargs: dict):
        """
        Parameters
        ----------
        Predictor
            Predictor to use in the committee.
        n_classifiers
            Number of logistic regression classifiers in the committee.
        kwargs
            Keyword arguments passed to the underlying Predictor.
        """
        self._committee = [Predictor(**kwargs)
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
            classifier.fit(features, labels)

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
            [classifier.predict(features)
             for classifier in self._committee],
            axis=1)
        assert predictions.shape == (features.shape[0], self.n_classifiers)
        return predictions


def AveragePredictions(predictor: Predictor) -> Predictor:
    """Wrapper for a predictor that averages predicted probabilities.

    Notes
    -----
    This effectively reduces the number of predictors to 1.

    Arguments
    ---------
    predictor
        Predictor to wrap.

    Returns
    -------
    Predictor
        Predictor with averaged predictions.
    """
    predictor.predict_ = predictor.predict

    def predict(features: numpy.ndarray) -> numpy.ndarray:
        predictions = predictor.predict_(features)
        return predictions.mean(axis=1).reshape((-1, 1))

    predictor.predict = predict

    return predictor
