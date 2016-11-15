"""Labeller classes."""

from abc import ABC, abstractmethod

import numpy


class Labeller(ABC):
    """Base class for labellers.

    Attributes
    ----------
    """

    @abstractmethod
    def query(self, id_: bytes) -> numpy.ndarray:
        """Queries the labeller.

        Parameters
        ----------
        id_
            ID of instance to label.

        Returns
        -------
        numpy.ndarray
            T x N x F label array.
        """
