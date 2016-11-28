"""Recommender classes."""

from abc import ABC, abstractmethod
from typing import Iterable, Sequence

import numpy


class Recommender(ABC):
    """Base class for recommenders.

    Attributes
    ----------
    """

    @abstractmethod
    def recommend(self, ids: Iterable[bytes],
                  predictions: numpy.ndarray) -> bytes:
        """Recommends an instance to label.

        Parameters
        ----------
        ids
            Iterable of IDs in the unlabelled data pool.
        predictions
            N x 1 array of predictions.

        Returns
        -------
        bytes
            ID of the instance to label.
        """


class RandomRecommender(Recommender):
    """Recommends instances at random."""

    def recommend(self, ids: Iterable[bytes],
                  predictions: numpy.ndarray) -> bytes:
        """Recommends an instance to label.

        Parameters
        ----------
        ids
            Iterable of IDs in the unlabelled data pool.
        predictions
            N x 1 array of predictions.

        Returns
        -------
        bytes
            ID of the instance to label.
        """
        return numpy.random.choice(list(ids))


class QBCRecommender(Recommender):
    """Recommends instances by committee disagreement."""

    def recommend(self, ids: Sequence[bytes],
                  predictions: numpy.ndarray) -> bytes:
        """Recommends an instance to label.

        Notes
        -----
        Assumes predictions are probabilities of positive binary label.

        Parameters
        ----------
        ids
            Sequence of IDs in the unlabelled data pool.
        predictions
            N x T array of predictions. The ith row must correspond with the ith
            ID in the sequence.

        Returns
        -------
        bytes
            ID of the instance to label.
        """
        assert predictions.shape[1] > 2, "QBC must have > 2 predictors."
        assert len(ids) == predictions.shape[0]
        labels = predictions >= 0.5
        n_agree = labels.sum(axis=1)
        agreement = numpy.abs(n_agree - labels.shape[1] / 2)
        return ids[agreement.argmin()]


# For safe string-based access to recommender classes.
RECOMMENDERS = {
    'RandomRecommender': RandomRecommender,
    'QBCRecommender': QBCRecommender,
}
