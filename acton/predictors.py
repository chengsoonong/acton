"""Predictor classes."""

from abc import ABC, abstractmethod
from typing import Iterable, Sequence

import acton.database
import acton.kde_predictor
import GPy as gpy
import numpy
import sklearn.base
import sklearn.linear_model
import sklearn.model_selection
import sklearn.preprocessing


class Predictor(ABC):
    """Base class for predictors.

    Attributes
    ----------
    prediction_type : str
        What kind of predictions this class generates, e.g. classification.s
    """
    prediction_type = 'classification'

    @abstractmethod
    def fit(self, ids: Iterable[int]):
        """Fits the predictor to labelled data.

        Parameters
        ----------
        ids
            List of IDs of instances to train from.
        """

    @abstractmethod
    def predict(self, ids: Sequence[int]) -> numpy.ndarray:
        """Predicts labels of instances.

        Notes
        -----
            Unlike in scikit-learn, predictions are always real-valued.
            Predicted labels for a classification problem are represented by
            predicted probabilities of each class.

        Parameters
        ----------
        ids
            List of IDs of instances to predict labels for.

        Returns
        -------
        numpy.ndarray
            An N x T x C array of corresponding predictions.
        """

    @abstractmethod
    def reference_predict(self, ids: Sequence[int]) -> numpy.ndarray:
        """Predicts labels using the best possible method.

        Parameters
        ----------
        ids
            List of IDs of instances to predict labels for.

        Returns
        -------
        numpy.ndarray
            An N x 1 x C array of corresponding predictions.
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

    def fit(self, ids: Iterable[int]):
        """Fits the predictor to labelled data.

        Parameters
        ----------
        ids
            List of IDs of instances to train from.
        """
        features = self._db.read_features(ids)
        labels = self._db.read_labels([0], ids)
        self._instance.fit(features, labels.ravel())

    def predict(self, ids: Sequence[int]) -> numpy.ndarray:
        """Predicts labels of instances.

        Notes
        -----
            Unlike in scikit-learn, predictions are always real-valued.
            Predicted labels for a classification problem are represented by
            predicted probabilities of each class.

        Parameters
        ----------
        ids
            List of IDs of instances to predict labels for.

        Returns
        -------
        numpy.ndarray
            An N x 1 x C array of corresponding predictions.
        """
        features = self._db.read_features(ids)
        try:
            probs = self._instance.predict_proba(features)
            return probs.reshape((probs.shape[0], 1, probs.shape[1]))
        except AttributeError:
            probs = self._instance.predict(features)
            if len(probs.shape) == 1:
                return probs.reshape((probs.shape[0], 1, 1))
            else:
                raise NotImplementedError()

    def reference_predict(self, ids: Sequence[int]) -> numpy.ndarray:
        """Predicts labels using the best possible method.

        Parameters
        ----------
        ids
            List of IDs of instances to predict labels for.

        Returns
        -------
        numpy.ndarray
            An N x 1 x C array of corresponding predictions.
        """
        return self.predict(ids)


def from_instance(predictor: sklearn.base.BaseEstimator,
                  db: acton.database.Database, regression: bool=False
                  ) -> Predictor:
    """Converts a scikit-learn predictor instance into a Predictor instance.

    Arguments
    ---------
    predictor
        scikit-learn predictor.
    db
        Database storing features and labels.
    regression
        Whether this predictor does regression (as opposed to classification).

    Returns
    -------
    Predictor
        Predictor instance wrapping the scikit-learn predictor.
    """
    ip = _InstancePredictor(predictor, db)
    if regression:
        ip.prediction_type = 'regression'
    return ip


def from_class(Predictor: type, regression: bool=False) -> type:
    """Converts a scikit-learn predictor class into a Predictor class.

    Arguments
    ---------
    Predictor
        scikit-learn predictor class.
    regression
        Whether this predictor does regression (as opposed to classification).

    Returns
    -------
    type
        Predictor class wrapping the scikit-learn class.
    """
    class Predictor_(_InstancePredictor):

        def __init__(self, db, **kwargs):
            super().__init__(instance=None, db=db)
            self._instance = Predictor(**kwargs)

    if regression:
        Predictor_.prediction_type = 'regression'

    return Predictor_


class Committee(Predictor):
    """A predictor using a committee of other predictors.

    Attributes
    ----------
    n_classifiers : int
        Number of logistic regression classifiers in the committee.
    subset_size : float
        Percentage of known labels to take subsets of to train the
        classifier. Lower numbers increase variety.
    _db : acton.database.Database
        Database storing features and labels.
    _committee : List[sklearn.linear_model.LogisticRegression]
        Underlying committee of logistic regression classifiers.
    _reference_predictor : Predictor
        Reference predictor trained on all known labels.
    """

    def __init__(self, Predictor: type, db: acton.database.Database,
                 n_classifiers: int=10, subset_size: float=0.6,
                 **kwargs: dict):
        """
        Parameters
        ----------
        Predictor
            Predictor to use in the committee.
        db
            Database storing features and labels.
        n_classifiers
            Number of logistic regression classifiers in the committee.
        subset_size
            Percentage of known labels to take subsets of to train the
            classifier. Lower numbers increase variety.
        kwargs
            Keyword arguments passed to the underlying Predictor.
        """
        self.n_classifiers = n_classifiers
        self.subset_size = subset_size
        self._db = db
        self._committee = [Predictor(db=db, **kwargs)
                           for _ in range(n_classifiers)]
        self._reference_predictor = Predictor(db=db, **kwargs)

    def fit(self, ids: Iterable[int]):
        """Fits the predictor to labelled data.

        Parameters
        ----------
        ids
            List of IDs of instances to train from.
        """
        # Get labels so we can stratify a split.
        labels = self._db.read_labels([0], ids)
        for classifier in self._committee:
            # Take a subsets to introduce variety.
            try:
                subset, _ = sklearn.model_selection.train_test_split(
                    ids, train_size=self.subset_size, stratify=labels)
            except ValueError:
                # Too few labels.
                subset = ids
            classifier.fit(subset)
        self._reference_predictor.fit(ids)

    def predict(self, ids: Sequence[int]) -> numpy.ndarray:
        """Predicts labels of instances.

        Notes
        -----
            Unlike in scikit-learn, predictions are always real-valued.
            Predicted labels for a classification problem are represented by
            predicted probabilities of each class.

        Parameters
        ----------
        ids
            List of IDs of instances to predict labels for.

        Returns
        -------
        numpy.ndarray
            An N x T x C array of corresponding predictions.
        """
        predictions = numpy.concatenate(
            [classifier.predict(ids)
             for classifier in self._committee],
            axis=1)
        assert predictions.shape[:2] == (len(ids), len(self._committee))
        return predictions

    def reference_predict(self, ids: Sequence[int]) -> numpy.ndarray:
        """Predicts labels using the best possible method.

        Parameters
        ----------
        ids
            List of IDs of instances to predict labels for.

        Returns
        -------
        numpy.ndarray
            An N x 1 x C array of corresponding predictions.
        """
        return self._reference_predictor.predict(ids)


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
        predictions = predictions.mean(axis=1)
        return predictions.reshape(
            (predictions.shape[0], 1, predictions.shape[1]))

    predictor.predict = predict

    return predictor


class GPClassifier(Predictor):
    """Classifier using Gaussian processes.

    Attributes
    ----------
    max_iters : int
        Maximum optimisation iterations.
    label_encoder : sklearn.preprocessing.LabelEncoder
        Encodes labels as integers.
    model_ : gpy.models.GPClassification
        GP model.
    _db : acton.database.Database
        Database storing features and labels.
    """
    def __init__(self, db: acton.database.Database, max_iters: int=50000,
                 n_jobs: int=1):
        """
        Parameters
        ----------
        db
            Database.
        max_iters
            Maximum optimisation iterations.
        n_jobs
            Does nothing; here for compatibility with sklearn.
        """
        self._db = db
        self.max_iters = max_iters

    def fit(self, ids: Iterable[int]):
        """Fits the predictor to labelled data.

        Parameters
        ----------
        ids
            List of IDs of instances to train from.
        """
        features = self._db.read_features(ids)
        labels = self._db.read_labels([0], ids).ravel()
        self.label_encoder_ = sklearn.preprocessing.LabelEncoder()
        labels = self.label_encoder_.fit_transform(labels).reshape((-1, 1))
        if len(self.label_encoder_.classes_) > 2:
            raise ValueError(
                'GPClassifier only supports binary classification.')
        self.model_ = gpy.models.GPClassification(features, labels)
        self.model_.optimize('bfgs', max_iters=self.max_iters)

    def predict(self, ids: Sequence[int]) -> numpy.ndarray:
        """Predicts labels of instances.

        Notes
        -----
            Unlike in scikit-learn, predictions are always real-valued.
            Predicted labels for a classification problem are represented by
            predicted probabilities of each class.

        Parameters
        ----------
        ids
            List of IDs of instances to predict labels for.

        Returns
        -------
        numpy.ndarray
            An N x 1 x C array of corresponding predictions.
        """
        features = self._db.read_features(ids)
        p_predictions = self.model_.predict(features)[0]
        n_predictions = 1 - p_predictions
        predictions = numpy.concatenate([n_predictions, p_predictions], axis=1)
        assert predictions.shape[1] == 2
        return predictions.reshape((-1, 1, 2))

    def reference_predict(self, ids: Sequence[int]) -> numpy.ndarray:
        """Predicts labels using the best possible method.

        Parameters
        ----------
        ids
            List of IDs of instances to predict labels for.

        Returns
        -------
        numpy.ndarray
            An N x 1 x C array of corresponding predictions.
        """
        return self.predict(ids)


# Helper functions to generate predictor classes.


def _logistic_regression() -> type:
    return from_class(sklearn.linear_model.LogisticRegression)


def _linear_regression() -> type:
    return from_class(sklearn.linear_model.LinearRegression, regression=True)


def _logistic_regression_committee() -> type:
    def make_committee(db, *args, **kwargs):
        return Committee(_logistic_regression(), db, *args, **kwargs)

    return make_committee


def _kde() -> type:
    return from_class(acton.kde_predictor.KDEClassifier)


PREDICTORS = {
    'LogisticRegression': _logistic_regression(),
    'LogisticRegressionCommittee': _logistic_regression_committee(),
    'LinearRegression': _linear_regression(),
    'KDE': _kde(),
    'GPC': GPClassifier,
}
