"""Main processing script for Acton."""

import contextlib
from typing import List

import acton.labellers
import astropy.io.ascii
import h5py


def guess_type(data_path: str):
    """Guesses the type of a file.

    Arguments
    ---------
    data_path
        Path to file.

    Returns
    -------
    object
        Context manager opener for the type.

    Raises
    ------
    ValueError
        Unknown type.
    """
    try:
        with h5py.File(data_path, 'r') as _:
            _
        return lambda path: h5py.File(path, 'r+')
    except OSError:
        pass

    try:
        astropy.io.ascii.read(data_path)

        def ascii(path):
            yield astropy.io.ascii.read(path)

        return contextlib.contextmanager(ascii)
    except ValueError:
        raise ValueError('Unknown file type: {}'.format(data_path))


def main(data_path: str, feature_cols: List[str], label_col: str,
         id_col: str=None):
    """
    Arguments
    ---------
    data_path
        Path to data file.
    feature_cols
        Column names of the features.
    label_col
        Column name of the labels.
    id_col
        Column name of the IDs. If not specified, IDs will be automatically
        assigned.
    """
    opener = guess_type(data_path)
    with opener(data_path) as f:
        features = f[feature_cols]
        labels = f[label_col]

        if id_col:
            ids = f[id_col]
            ids = [str(id_).encode('ascii') for id_ in ids]
        else:
            ids = [str(i).encode('ascii') for i in range(len(labels))]

        # Label 10 instances.
        labeller = acton.labellers.ASCIITableLabeller()