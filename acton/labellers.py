"""Labeller classes."""

from abc import ABC, abstractmethod

import acton.database
import astropy.io.ascii
import numpy


class Labeller(ABC):
    """Base class for labellers.

    Attributes
    ----------
    """

    @abstractmethod
    def query(self, id_: int) -> numpy.ndarray:
        """Queries the labeller.

        Parameters
        ----------
        id_
            ID of instance to label.

        Returns
        -------
        numpy.ndarray
            T x F label array.
        """


class ASCIITableLabeller(Labeller):
    """Labeller that obtains labels from an ASCII table.

    Attributes
    ----------
    path : str
        Path to table.
    id_col : str
        Name of the column where IDs are stored.
    label_col : str
        Name of the column where binary labels are stored.
    _table : astropy.table.Table
        Table object.
    """

    def __init__(self, path: str, id_col: str, label_col: str):
        """
        path
            Path to table.
        id_col
            Name of the column where IDs are stored.
        label_col
            Name of the column where binary labels are stored.
        """
        self.path = path
        self.id_col = id_col
        self.label_col = label_col
        self._table = astropy.io.ascii.read(self.path)
        self._id_to_name = {}
        for id_, row in enumerate(self._table):
            name = row[self.id_col]
            self._id_to_name[id_] = name

    def query(self, id_: int) -> numpy.ndarray:
        """Queries the labeller.

        Parameters
        ----------
        id_
            ID of instance to label.

        Returns
        -------
        numpy.ndarray
            1 x 1 label array.
        """
        for row in self._table:
            if row[self.id_col] == self._id_to_name[id_]:
                return row[self.label_col].reshape((1, 1))
        raise KeyError('Unknown id: {}'.format(id_))


class DatabaseLabeller(Labeller):
    """Labeller that obtains labels from a Database.

    Attributes
    ----------
    _db : acton.database.Database
        Database with labels.
    """

    def __init__(self, db: acton.database.Database):
        """
        db
            Database with labels to read from.
        """
        self._db = db

    def query(self, id_: int) -> numpy.ndarray:
        """Queries the labeller.

        Parameters
        ----------
        id_
            ID of instance to label.

        Returns
        -------
        numpy.ndarray
            1 x 1 label array.
        """
        return self._db.read_labels([0], [id_]).reshape((1, 1))


# For safe string-based access to labeller classes.
LABELLERS = {
    'ASCIITableLabeller': ASCIITableLabeller,
    'DatabaseLabeller': DatabaseLabeller,
}
