"""Main processing script for Acton."""

import logging
import os.path
import tempfile
from typing import Iterable, List, TypeVar

import acton.database
import acton.labellers
import acton.predictors
import acton.recommenders
import astropy.io.ascii as io_ascii
import astropy.table
import h5py
import matplotlib.pyplot as plt
import numpy
import sklearn.cross_validation
import sklearn.metrics

T = TypeVar('T')


def product(seq: Iterable[int]):
    """Finds the product of a list of ints.

    Arguments
    ---------
    seq
        List of ints.

    Returns
    -------
    int
        Product.
    """
    prod = 1
    for i in seq:
        prod *= i
    return prod


def draw(n: int, lst: List[T], replace: bool=True) -> List[T]:
    """Draws n random elements from a list.

    Arguments
    ---------
    n
        Number of elements to draw.
    lst
        List of elements to draw from.
    replace
        Draw with replacement.

    Returns
    -------
    List[T]
        n random elements.
    """
    # While we use replace=False generally in this codebase, the NumPy default
    # is True - so we should use that here.
    return list(numpy.random.choice(lst, size=n, replace=replace))


def simulate_active_learning(
        ids: Iterable[bytes],
        db: acton.database.Database,
        n_initial_labels: int=10,
        n_epochs: int=10,
        test_size: int=0.2,
        predictor='LogisticRegression',
        recommender='RandomRecommender'):
    """Simulates an active learning task.

    arguments
    ---------
    ids
        IDs of instances in the unlabelled pool.
    db
        Database with features and labels.
    n_initial_labels
        Number of initial labels to draw.
    n_epochs
        Number of epochs.
    test_size
        Percentage size of testing set.
    predictor
        Name of predictor to make predictions.
    recommender
        Name of recommender to make recommendations.
    """
    # Validation.
    if predictor not in acton.predictors.PREDICTORS:
        raise ValueError('Unknown predictor: {}. Predictors are one of '
                         '{}.'.format(predictor,
                                      acton.predictors.PREDICTORS.keys()))
    if recommender not in acton.recommenders.RECOMMENDERS:
        raise ValueError('Unknown recommender: {}. Recommenders are one of '
                         '{}.'.format(recommender,
                                      acton.recommenders.RECOMMENDERS.keys()))

    # Split into training and testing sets.
    train_ids, test_ids = sklearn.cross_validation.train_test_split(
        ids, test_size=test_size)
    test_features = db.read_features(test_ids)
    test_labels = db.read_labels([b'0'], test_ids)

    # Set up predictor, labeller, and recommender.
    # TODO(MatthewJA): Handle multiple labellers better than just averaging.
    predictor = acton.predictors.AveragePredictions(
        acton.predictors.PREDICTORS[predictor]()
    )
    labeller = acton.labellers.DatabaseLabeller(db)
    recommender = acton.recommenders.RECOMMENDERS[recommender]()

    # Draw some initial labels.
    recommendations = draw(n_initial_labels, train_ids, replace=False)
    logging.debug('Recommending: {}'.format(recommendations))

    # This will store all IDs of things we have already labelled.
    labelled_ids = []
    # This will store all the corresponding labels.
    labels = numpy.zeros((0, 1))

    # Simulation loop.
    accuracies = []  # List of (n_labels, accuracy) pairs.
    for epoch in range(n_epochs):
        # Label the recommendations.
        new_labels = numpy.array([
            labeller.query(id_) for id_ in recommendations]).reshape((-1, 1))
        labelled_ids.extend(recommendations)
        labels = numpy.concatenate([labels, new_labels], axis=0)

        # Pass the labels to the predictor.
        predictor.fit(db.read_features(labelled_ids), labels)

        # Evaluate the predictor.
        test_pred = predictor.predict(test_features)
        accuracy = sklearn.metrics.accuracy_score(
            test_labels.ravel(), test_pred.round().ravel())
        accuracies.append((len(labelled_ids), accuracy))
        logging.debug('Accuracy: {}'.format(accuracy))

        # Pass the predictions to the recommender.
        predictions = predictor.predict(db.read_features(ids))
        recommendations = [recommender.recommend(ids, predictions)]
        logging.debug('Recommending: {}'.format(recommendations))

    plt.plot(*zip(*accuracies))
    plt.show()


def db_from_ascii(
        db: acton.database.Database,
        data: astropy.table.Table,
        feature_cols: List[str],
        label_col: str,
        ids: List[bytes],
        id_col: str=None):
    """Reads an ASCII table into a database.

    Notes
    -----
    The entire file is copied into memory.

    Arguments
    ---------
    db
        Database.
    data
        ASCII table.
    feature_cols
        List of column names of the features. If empty, all non-label and non-ID
        columns will be used.
    label_col
        Column name of the labels.
    id_col
        Column name of the IDs.
    ids
        List of instance IDs.
    """
    # Read in features.
    columns = data.keys()
    if not feature_cols:
        # If there are no features given, use all columns.
        feature_cols = [c for c in columns
                        if c != label_col and c != id_col]

    # This converts the features from a table to an array.
    features = data[feature_cols].as_array()
    features = features.view(numpy.float64).reshape(features.shape + (-1,))

    # Read in labels.
    labels = numpy.array(data[label_col], dtype=bool).reshape((1, -1, 1))

    # We want to support multiple labellers in the future, but currently don't.
    # So every labeller is the same, ID = 0.
    labeller_ids = [b'0']

    # Write to database.
    db.write_features(ids, features)
    db.write_labels(labeller_ids, ids, labels)


def db_from_hdf5(
        db: acton.database.Database,
        data: h5py.File,
        feature_cols: List[str],
        label_col: str,
        ids: List[bytes],
        id_col: str=None,
        batch_size: int=5000):
    """Reads a HDF5 file into a database.

    Arguments
    ---------
    db
        Database.
    data
        HDF5 file.
    feature_cols
        List of column names of the features. If empty, all non-label and non-ID
        columns will be used.
    label_col
        Column name of the labels.
    ids
        List of instance IDs.
    id_col
        Column name of the IDs.
    batch_size
        Number of instances to read at once.
    """
    if not feature_cols:
        raise ValueError('Must specify feature columns for HDF5.')

    # Validation. If you pass in a feature that maps to a multidimensional
    # table, you must have only passed in one feature.
    is_multidimensional = any(len(data[f_col].shape) > 1 or
                              not product(data[f_col].shape[1:]) == 1
                              for f_col in feature_cols)
    if is_multidimensional and len(feature_cols) != 1:
        raise ValueError('Feature arrays and feature columns cannot be mixed. '
                         'To read in features from a multidimensional dataset, '
                         'only specify one feature column name.')

    # We want to support multiple labellers in the future, but currently don't.
    # So every labeller is the same, ID = 0.
    labeller_ids = [b'0']

    if is_multidimensional:
        n_left = data[feature_cols[0]].shape[0]
        for idx in range(0, n_left, batch_size):
            id_batch = ids[idx:idx + batch_size]
            features_batch = data[feature_cols[0]][idx:idx + batch_size, :]
            logging.debug('Writing features to database. ({:.02%})'.format(
                (idx + features_batch.shape[0]) / n_left))
            label_batch = data[label_col][idx:idx + batch_size]
            label_batch = label_batch.reshape((1, -1, 1))
            db.write_features(id_batch, features_batch)
            db.write_labels(labeller_ids, id_batch, label_batch)
    else:
        n_features = len(feature_cols)
        features_batch = numpy.zeros((batch_size, n_features))

        n_left = data[feature_cols[0]].shape[0]
        for idx in range(0, n_left, batch_size):
            # :feature_batch.shape[0] should handle cases where n_left is not
            # divisible by batch_size (without accidentally writing some zero
            # vectors).
            batch_length = None
            for feature_idx, feature_col in enumerate(feature_cols):
                batch_length = batch_length or feature_batch.shape[0]
                assert batch_length == feature_batch.shape[0]
                feature_batch = data[feature_col][idx:idx + batch_size]
                features_batch[:batch_length, :] = feature_batch
            label_batch = data[label_col][idx:idx + batch_size]
            id_batch = ids[idx:idx + batch_size]
            db.write_features(id_batch, features_batch[:batch_length])
            db.write_labels(labeller_ids, id_batch, label_batch)


def main(data_path: str, feature_cols: List[str], label_col: str,
         id_col: str=None, n_epochs: int=10, initial_count: int=10,
         predictor: str='LogisticRegression',
         recommender: str='RandomRecommender'):
    """
    Arguments
    ---------
    data_path
        Path to data file.
    feature_cols
        List of column names of the features. If empty, all non-label and non-ID
        columns will be used.
    label_col
        Column name of the labels.
    id_col
        Column name of the IDs. If not specified, IDs will be automatically
        assigned.
    n_epochs
        Number of epochs to run.
    initial_count
        Number of random instances to label initially.
    predictor
        Name of predictor to make predictions.
    recommender
        Name of recommender to make recommendations.
    """
    # Read in the features, labels, and IDs. This is different for HDF5 and
    # ASCII files.
    is_ascii = not data_path.endswith('.h5')
    with tempfile.TemporaryDirectory(prefix='acton') as tempdir:
        temp_db_filename = os.path.join(tempdir, 'db.h5')

        # First, find the maximum ID length. Do this by reading in IDs.
        if is_ascii:
            data = io_ascii.read(data_path)
            if id_col:
                ids = [str(id_).encode('utf-8') for id_ in data[id_col]]
            else:
                ids = [str(id_).encode('utf-8')
                       for id_ in range(len(data[label_col]))]
        else:  # HDF5
            with h5py.File(data_path, 'r+') as f_h5:
                if id_col:
                    ids = [str(id_).encode('utf-8') for id_ in f_h5[id_col]]
                else:
                    ids = [str(id_).encode('utf-8')
                           for id_ in range(f_h5[label_col].shape[0])]

        max_id_length = max(len(id_) for id_ in ids)

        with acton.database.HDF5Database(
                temp_db_filename,
                max_id_length=max_id_length,
                label_dtype='bool',
                feature_dtype='float64') as db:
            if is_ascii:  # ASCII
                db_from_ascii(
                    db, io_ascii.read(data_path), feature_cols, label_col,
                    ids, id_col)
            else:  # HDF5
                with h5py.File(data_path, 'r+') as f_h5:
                    db_from_hdf5(db, f_h5, feature_cols, label_col, ids, id_col)

            # Simulate the active learning task.
            simulate_active_learning(ids, db, n_epochs=n_epochs,
                                     n_initial_labels=initial_count,
                                     predictor=predictor,
                                     recommender=recommender)
