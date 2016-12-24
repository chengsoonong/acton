"""Recommender classes."""

from abc import ABC, abstractmethod
import logging
from typing import Iterable, Sequence

import acton.database
import numpy
import scipy.spatial


def mmr_choose(features: numpy.ndarray, scores: numpy.ndarray, n: int,
               l: float=0.5) -> Iterable[int]:
    """Chooses n scores using maximal marginal relevance.

    Notes
    -----
    Scores are chosen from highest to lowest. If there are less scores to choose
    from than requested, all scores will be returned in order of preference.

    Parameters
    ----------
    scores
        1D array of scores.
    n
        Number of scores to choose.
    l
        Lambda parameter for MMR. l = 1 gives a relevance-ranked list and l = 0
        gives a maximal diversity ranking.

    Returns
    -------
    Iterable[int]
        List of indices of scores chosen.
    """
    if n < 0:
        raise ValueError('n must be a non-negative integer.')

    if n == 0:
        return []

    selections = [scores.argmax()]
    selections_set = set(selections)

    logging.debug('Running MMR.')
    dists = []
    dists_matrix = None
    while len(selections) < n:
        if len(selections) % (n // 10) == 0:
            logging.debug('MMR epoch {}/{}.'.format(len(selections), n))
        # Compute distances for last selection.
        last = features[selections[-1]:selections[-1] + 1]
        last_dists = numpy.linalg.norm(features - last, axis=1)
        dists.append(last_dists)
        dists_matrix = numpy.array(dists)

        next_best = None
        next_best_margin = float('-inf')

        for i in range(len(scores)):
            if i in selections_set:
                continue

            margin = l * (scores[i] - (1 - l) * dists_matrix[:, i].max())
            if margin > next_best_margin:
                next_best_margin = margin
                next_best = i

        if next_best is None:
            break

        selections.append(next_best)
        selections_set.add(next_best)

    return selections


class Recommender(ABC):
    """Base class for recommenders.

    Attributes
    ----------
    """

    @abstractmethod
    def recommend(self, ids: Iterable[int],
                  predictions: numpy.ndarray,
                  n: int=1, diversity: float=0.5) -> Sequence[int]:
        """Recommends an instance to label.

        Parameters
        ----------
        ids
            Iterable of IDs in the unlabelled data pool.
        predictions
            N x 1 array of predictions.
        n
            Number of recommendations to make.
        diversity
            Recommendation diversity in [0, 1].

        Returns
        -------
        Sequence[int]
            IDs of the instances to label.
        """


class RandomRecommender(Recommender):
    """Recommends instances at random."""

    def __init__(self, db: acton.database.Database):
        """
        Parameters
        ----------
        db
            Features database.
        """
        self._db = db

    def recommend(self, ids: Iterable[int],
                  predictions: numpy.ndarray,
                  n: int=1, diversity: float=0.5) -> Sequence[int]:
        """Recommends an instance to label.

        Parameters
        ----------
        ids
            Iterable of IDs in the unlabelled data pool.
        predictions
            N x 1 array of predictions.
        n
            Number of recommendations to make.
        diversity
            Recommendation diversity in [0, 1].

        Returns
        -------
        Sequence[int]
            IDs of the instances to label.
        """
        return numpy.random.choice(list(ids), size=n)


class QBCRecommender(Recommender):
    """Recommends instances by committee disagreement."""

    def __init__(self, db: acton.database.Database):
        """
        Parameters
        ----------
        db
            Features database.
        """
        self._db = db

    def recommend(self, ids: Iterable[int],
                  predictions: numpy.ndarray,
                  n: int=1, diversity: float=0.5) -> Sequence[int]:
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
        n
            Number of recommendations to make.
        diversity
            Recommendation diversity in [0, 1].

        Returns
        -------
        Sequence[int]
            IDs of the instances to label.
        """
        assert predictions.shape[1] > 2, "QBC must have > 2 predictors."
        assert len(ids) == predictions.shape[0]
        assert 0 <= diversity <= 1
        labels = predictions >= 0.5
        n_agree = labels.sum(axis=1)
        disagreement = labels.shape[1] - numpy.abs(
            n_agree - labels.shape[1] / 2)

        # MMR
        indices = mmr_choose(self._db.read_features(ids), disagreement, n,
                             l=diversity)
        return [ids[i] for i in indices]


class UncertaintyRecommender(Recommender):
    """Recommends instances by uncertainty sampling."""

    def __init__(self, db: acton.database.Database):
        """
        Parameters
        ----------
        db
            Features database.
        """
        self._db = db

    def recommend(self, ids: Iterable[int],
                  predictions: numpy.ndarray,
                  n: int=1, diversity: float=0.5) -> Sequence[int]:
        """Recommends an instance to label.

        Notes
        -----
        Assumes predictions are probabilities of positive binary label.

        Parameters
        ----------
        ids
            Sequence of IDs in the unlabelled data pool.
        predictions
            N x 1 array of predictions. The ith row must correspond with the ith
            ID in the sequence.
        n
            Number of recommendations to make.
        diversity
            Recommendation diversity in [0, 1].

        Returns
        -------
        Sequence[int]
            IDs of the instances to label.
        """
        if predictions.shape[1] != 1:
            raise ValueError('Uncertainty sampling must have one predictor')

        assert len(ids) == predictions.shape[0]

        proximities = 0.5 - numpy.abs(predictions - 0.5)
        indices = mmr_choose(self._db.read_features(ids), proximities, n,
                             l=diversity)
        return [ids[i] for i in indices]


# For safe string-based access to recommender classes.
RECOMMENDERS = {
    'RandomRecommender': RandomRecommender,
    'QBCRecommender': QBCRecommender,
    'UncertaintyRecommender': UncertaintyRecommender,
}
