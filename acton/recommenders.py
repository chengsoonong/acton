"""Recommender classes."""

from abc import ABC, abstractmethod
from typing import Iterable

import numpy


class Recommender(ABC):
    """Base class for recommenders.

    Attributes
    ----------
    """

    @abstractmethod
    def recommend(self, ids: Iterable[bytes]) -> bytes:
        """Recommends an instance to label.

        Parameters
        ----------
        ids
            Iterable of IDs in the unlabelled data pool.

        Returns
        -------
        bytes
            ID of the instance to label.
        """


class RandomRecommender(Recommender):
    """Recommends instances at random."""

    def recommend(self, ids: Iterable[bytes]) -> bytes:
        """Recommends an instance to label.

        Parameters
        ----------
        ids
            Iterable of IDs in the unlabelled data pool.

        Returns
        -------
        bytes
            ID of the instance to label.
        """
        return numpy.random.choice(list(ids))


# For safe string-based access to recommender classes.
RECOMMENDERS = {
    'RandomRecommender': RandomRecommender,
}
