"""Predictor classes."""

from abc import ABC, abstractmethod
from typing import List

import acton.database
import numpy
import sklearn.base


class Predictor(ABC):
    """Base class for predictors.

    Attributes
    ----------
    """

    @abstractmethod
    def fit(self, ids: List[bytes]):
        """Fits the predictor to labelled data.

        Parameters
        ----------
        ids
            List of IDs of instances to train from.
        """

    @abstractmethod
    def predict(self, ids: List[bytes]) -> numpy.ndarray:
        """Predicts labels of instances.

        Notes
        -----
            Unlike in scikit-learn, predictions are always real-valued.
            Predicted labels for a classification problem are represented by
            predicted probabilities of the positive class.

        Parameters
        ----------
        ids
            List of IDs of instances to predict labels for.

        Returns
        -------
        numpy.ndarray
            An N x T array of corresponding predictions.
        """


class _InstancePredictor(Predictor):
    """Wrapper for a scikit-learn instance.

    Attributes
    ----------
    _db : acton.database.Database
        Database storing features and labels.
    _instance : sklearn.base.BaseEstimator
        scikit-learn predictor instance.
    """

    def __init__(self, instance: sklearn.base.BaseEstimator,
                 db: acton.database.Database):
        """
        Arguments
        ---------
        instance
            scikit-learn predictor instance.
        db
            Database storing features and labels.
        """
        self._db = db
        self._instance = instance

    def fit(self, ids: List[bytes]):
        """Fits the predictor to labelled data.

        Parameters
        ----------
        ids
            List of IDs of instances to train from.
        """
        features = self._db.read_features(ids)
        labels = self._db.read_labels([b'0'], ids)
        self._instance.fit(features, labels.ravel())

    def predict(self, ids: List[bytes]) -> numpy.ndarray:
        """Predicts labels of instances.

        Notes
        -----
            Unlike in scikit-learn, predictions are always real-valued.
            Predicted labels for a classification problem are represented by
            predicted probabilities of the positive class.

        Parameters
        ----------
        ids
            List of IDs of instances to predict labels for.

        Returns
        -------
        numpy.ndarray
            An N x T array of corresponding predictions.
        """
        features = self._db.read_features(ids)
        return self._instance.predict_proba(features)[:, 1:]


def from_instance(predictor: sklearn.base.BaseEstimator,
                  db: acton.database.Database) -> Predictor:
    """Converts a scikit-learn predictor instance into a Predictor instance.

    Arguments
    ---------
    predictor
        scikit-learn predictor.
    db
        Database storing features and labels.

    Returns
    -------
    Predictor
        Predictor instance wrapping the scikit-learn predictor.
    """
    return _InstancePredictor(predictor, db)


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

        def __init__(self, db, **kwargs):
            super().__init__(instance=None, db=db)
            self._instance = Predictor(**kwargs)

    return Predictor_


class Committee(Predictor):
    """A predictor using a committee of other predictors.

    Attributes
    ----------
    n_classifiers : int
        Number of logistic regression classifiers in the committee.
    _db : acton.database.Database
        Database storing features and labels.
    _committee : List[sklearn.linear_model.LogisticRegression]
        Underlying committee of logistic regression classifiers.
    """

    def __init__(self, Predictor: type, db: acton.database.Database,
                 n_classifiers: int=10, **kwargs: dict):
        """
        Parameters
        ----------
        Predictor
            Predictor to use in the committee.
        db
            Database storing features and labels.
        n_classifiers
            Number of logistic regression classifiers in the committee.
        kwargs
            Keyword arguments passed to the underlying Predictor.
        """
        self.n_classifiers = n_classifiers
        self._db = db
        self._committee = [Predictor(db=db, **kwargs)
                           for _ in range(n_classifiers)]

    def fit(self, ids: List[bytes]):
        """Fits the predictor to labelled data.

        Parameters
        ----------
        ids
            List of IDs of instances to train from.
        """
        # TODO(MatthewJA): Introduce committee variety.
        for classifier in self._committee:
            classifier.fit(ids)

    def predict(self, ids: List[bytes]) -> numpy.ndarray:
        """Predicts labels of instances.

        Notes
        -----
            Unlike in scikit-learn, predictions are always real-valued.
            Predicted labels for a classification problem are represented by
            predicted probabilities of the positive class.

        Parameters
        ----------
        ids
            List of IDs of instances to predict labels for.

        Returns
        -------
        numpy.ndarray
            An N x T array of corresponding predictions.
        """
        predictions = numpy.concatenate(
            [classifier.predict(ids)
             for classifier in self._committee],
            axis=1)
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
