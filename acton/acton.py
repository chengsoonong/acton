"""Main processing script for Acton."""

import logging
import os.path
import tempfile
from typing import Iterable, List, TypeVar

import acton.database
import acton.labellers
import acton.predictors
import acton.proto.io
import acton.proto.wrappers
import acton.recommenders
import astropy.io.ascii as io_ascii
import astropy.table
import numpy
import sklearn.cross_validation
import sklearn.linear_model
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
        output_path: str,
        n_initial_labels: int=10,
        n_epochs: int=10,
        test_size: int=0.2,
        recommender: str='RandomRecommender',
        predictor: str='LogisticRegression'):
    """Simulates an active learning task.

    arguments
    ---------
    ids
        IDs of instances in the unlabelled pool.
    db
        Database with features and labels.
    output_path
        Path to output intermediate predictions to. Will be overwritten.
    n_initial_labels
        Number of initial labels to draw.
    n_epochs
        Number of epochs.
    test_size
        Percentage size of testing set.
    recommender
        Name of recommender to make recommendations.
    predictor
        Name of predictor to make predictions.
    """
    # Validation.
    if recommender not in acton.recommenders.RECOMMENDERS:
        raise ValueError('Unknown recommender: {}. Recommenders are one of '
                         '{}.'.format(recommender,
                                      acton.recommenders.RECOMMENDERS.keys()))

    if predictor not in acton.predictors.PREDICTORS:
        raise ValueError('Unknown predictor: {}. predictors are one of '
                         '{}.'.format(predictor,
                                      acton.predictors.PREDICTORS.keys()))

    # Split into training and testing sets.
    train_ids, test_ids = sklearn.cross_validation.train_test_split(
        ids, test_size=test_size)
    test_labels = db.read_labels([b'0'], test_ids)

    # Set up predictor, labeller, and recommender.
    # TODO(MatthewJA): Handle multiple labellers better than just averaging.
    predictor_name = predictor  # For saving.
    predictor = acton.predictors.PREDICTORS[predictor](db=db)

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
    logging.debug('Writing protobufs to {}.'.format(output_path))
    writer = acton.proto.io.write_protos(output_path)
    next(writer)  # Prime the coroutine.
    for epoch in range(n_epochs):
        logging.info('Epoch {}/{}'.format(epoch + 1, n_epochs))
        # Label the recommendations.
        new_labels = numpy.array([
            labeller.query(id_) for id_ in recommendations]).reshape((-1, 1))
        labelled_ids.extend(recommendations)
        labels = numpy.concatenate([labels, new_labels], axis=0)

        # Here, we would write the labels to the database, but they're already
        # there since we're just reading them from there anyway.
        pass

        # Pass the labels to the predictor.
        predictor.fit(labelled_ids)

        # Evaluate the predictor.
        test_pred = predictor.reference_predict(test_ids)
        accuracy = sklearn.metrics.accuracy_score(
            test_labels.ravel(), test_pred.mean(axis=1).round().ravel())
        accuracies.append((len(labelled_ids), accuracy))
        logging.debug('Accuracy: {}'.format(accuracy))

        # Construct a protobuf for outputting predictions.
        proto = acton.proto.wrappers.from_predictions(
            test_ids,
            test_pred.reshape((1,) + test_pred.shape),
            predictor=predictor_name,
            db_path=db.path,
            db_class=db.__class__.__name__)
        # Then write them to a file.
        writer.send(proto.proto)

        # Pass the predictions to the recommender.
        unlabelled_ids = list(set(ids) - set(labelled_ids))
        predictions = predictor.predict(unlabelled_ids)
        recommendations = [
            recommender.recommend(unlabelled_ids,
                                  predictions)
        ]
        logging.debug('Recommending: {}'.format(recommendations))


def main(data_path: str, feature_cols: List[str], label_col: str,
         output_path: str, id_col: str=None, n_epochs: int=10,
         initial_count: int=10, recommender: str='RandomRecommender',
         predictor: str='LogisticRegression'):
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
    output_path
        Path to output file. Will be overwritten.
    id_col
        Column name of the IDs. If not specified, IDs will be automatically
        assigned.
    n_epochs
        Number of epochs to run.
    initial_count
        Number of random instances to label initially.
    recommender
        Name of recommender to make recommendations.
    predictor
        Name of predictor to make predictions.
    """
    is_ascii = not data_path.endswith('.h5')
    if is_ascii:
        with acton.database.ASCIIReader(
                data_path, feature_cols=feature_cols, label_col=label_col,
                id_col=id_col) as reader:
            simulate_active_learning(reader.get_known_instance_ids(), reader,
                                     output_path,
                                     n_epochs=n_epochs,
                                     n_initial_labels=initial_count,
                                     recommender=recommender,
                                     predictor=predictor)

    else:
        # Assume HDF5.
        with acton.database.HDF5Reader(
                data_path, feature_cols=feature_cols, label_col=label_col,
                id_col=id_col) as reader:
            simulate_active_learning(reader.get_known_instance_ids(), reader,
                                     output_path,
                                     n_epochs=n_epochs,
                                     n_initial_labels=initial_count,
                                     recommender=recommender,
                                     predictor=predictor)
