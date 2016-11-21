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
import matplotlib.pyplot as plt
import numpy
import sklearn.cross_validation
import sklearn.metrics

T = TypeVar('T')


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
        test_size: int=0.2):
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
    """
    # Split into training and testing sets.
    train_ids, test_ids = sklearn.cross_validation.train_test_split(
        ids, test_size=test_size)
    test_features = db.read_features(test_ids)
    test_labels = db.read_labels([b'0'], test_ids)

    # Set up predictor, labeller, and recommender.
    predictor = acton.predictors.LogisticRegression()
    labeller = acton.labellers.DatabaseLabeller(db)
    recommender = acton.recommenders.RandomRecommender()

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


def main(data_path: str, label_col: str, id_col: str=None, n_epochs: int=10):
    """
    Arguments
    ---------
    data_path
        Path to data file.
    label_col
        Column name of the labels.
    id_col
        Column name of the IDs. If not specified, IDs will be automatically
        assigned.
    n_epochs
        Number of epochs to run.
    """
    # Read in the features, labels, and IDs.
    data = io_ascii.read(data_path)
    columns = data.keys()
    feature_cols = [c for c in columns if c != label_col]

    # This converts the features from a table to an array.
    features = data[feature_cols].as_array()
    features = features.view(numpy.float64).reshape(features.shape + (-1,))

    # Need to repeat for labels.
    labels = numpy.array(data[label_col], dtype=bool).reshape((1, -1, 1))

    # Then read in the IDs and compute their maximum length.
    if id_col:
        ids = [str(id_).encode('utf-8') for id_ in data[id_col]]
    else:
        ids = [str(id_).encode('utf-8') for id_ in range(labels.shape[1])]
    max_id_length = max(len(id_) for id_ in ids)

    # We want to support multiple labellers in the future, but currently don't.
    # So every labeller is the same, ID = 0.
    labeller_ids = [b'0']

    # Store everything in a database. There's probably a better way to do this,
    # but it's a nice way to make sure our data formats are consistent.
    with tempfile.TemporaryDirectory(prefix='acton') as tempdir:
        temp_db_filename = os.path.join(tempdir, 'db.h5')
        with acton.database.HDF5Database(
                temp_db_filename,
                max_id_length=max_id_length,
                label_dtype='bool',
                feature_dtype='float64') as db:
            db.write_features(ids, features)
            db.write_labels(labeller_ids, ids, labels)

            # Simulate the active learning task.
            simulate_active_learning(ids, db, n_epochs=n_epochs)
