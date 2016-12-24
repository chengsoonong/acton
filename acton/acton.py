"""Main processing script for Acton."""

import logging
from typing import Iterable, List, TypeVar

import acton.database
import acton.labellers
import acton.predictors
import acton.proto.io
import acton.proto.wrappers
import acton.recommenders
import numpy
import pandas
import sklearn.cross_validation
import sklearn.linear_model
import sklearn.metrics

T = TypeVar('T')


def draw(n: int, lst: List[T], replace: bool=True) -> List[T]:
    """Draws n random elements from a list.

    Parameters
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
        ids: Iterable[int],
        db: acton.database.Database,
        db_kwargs: dict,
        output_path: str,
        n_initial_labels: int=10,
        n_epochs: int=10,
        test_size: int=0.2,
        recommender: str='RandomRecommender',
        predictor: str='LogisticRegression',
        n_recommendations: int=1):
    """Simulates an active learning task.

    Parameters
    ---------
    ids
        IDs of instances in the unlabelled pool.
    db
        Database with features and labels.
    db_kwargs
        Keyword arguments for the database constructor.
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
    n_recommendations
        Number of recommendations to make at once.
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
    test_ids.sort()

    # Set up predictor, labeller, and recommender.
    # TODO(MatthewJA): Handle multiple labellers better than just averaging.
    predictor_name = predictor  # For saving.
    predictor = acton.predictors.PREDICTORS[predictor](db=db)

    labeller = acton.labellers.DatabaseLabeller(db)
    recommender = acton.recommenders.RECOMMENDERS[recommender](db=db)

    # Draw some initial labels.
    recommendations = draw(n_initial_labels, train_ids, replace=False)
    logging.debug('Recommending: {}'.format(recommendations))

    # This will store all IDs of things we have already labelled.
    labelled_ids = []
    # This will store all the corresponding labels.
    labels = numpy.zeros((0, 1))

    # Simulation loop.
    logging.debug('Writing protobufs to {}.'.format(output_path))
    writer = acton.proto.io.write_protos(output_path)
    next(writer)  # Prime the coroutine.
    for epoch in range(n_epochs):
        logging.info('Epoch {}/{}'.format(epoch + 1, n_epochs))
        # Label the recommendations.
        new_labels = numpy.array([
            labeller.query(id_) for id_ in recommendations]).reshape((-1, 1))
        labelled_ids.extend(recommendations)
        labelled_ids.sort()
        labels = numpy.concatenate([labels, new_labels], axis=0)

        # Here, we would write the labels to the database, but they're already
        # there since we're just reading them from there anyway.
        pass

        # Pass the labels to the predictor.
        predictor.fit(labelled_ids)

        # Evaluate the predictor.
        test_pred = predictor.reference_predict(test_ids)

        # Construct a protobuf for outputting predictions.
        proto = acton.proto.wrappers.from_predictions(
            test_ids,
            test_pred.reshape((1,) + test_pred.shape),
            predictor=predictor_name,
            db_path=db.path,
            db_class=db.__class__.__name__,
            db_kwargs=db_kwargs)
        # Then write them to a file.
        writer.send(proto.proto)

        # Pass the predictions to the recommender.
        unlabelled_ids = list(set(ids) - set(labelled_ids))
        if not unlabelled_ids:
            logging.info('Labelled all instances.')
            break

        predictions = predictor.predict(unlabelled_ids)
        recommendations = recommender.recommend(
            unlabelled_ids, predictions, n=n_recommendations)
        logging.debug('Recommending: {}'.format(recommendations))


def try_pandas(data_path: str) -> bool:
    """Guesses if a file is a pandas file.

    Parameters
    ----------
    data_path
        Path to file.

    Returns
    -------
    bool
        True if the file is pandas.
    """
    try:
        pandas.read_hdf(data_path)
    except ValueError:
        return False

    return True


def main(data_path: str, feature_cols: List[str], label_col: str,
         output_path: str, n_epochs: int=10, initial_count: int=10,
         recommender: str='RandomRecommender',
         predictor: str='LogisticRegression', pandas_key: str='',
         n_recommendations: int=1):
    """
    Parameters
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
    n_epochs
        Number of epochs to run.
    initial_count
        Number of random instances to label initially.
    recommender
        Name of recommender to make recommendations.
    predictor
        Name of predictor to make predictions.
    pandas_key
        Key for pandas HDF5. Specify iff using pandas.
    n_recommendations
        Number of recommendations to make at once.
    """
    db_kwargs = {
        'feature_cols': feature_cols,
        'label_col': label_col,
    }

    is_ascii = not data_path.endswith('.h5')
    if is_ascii:
        logging.debug('Reading {} as ASCII.'.format(data_path))
        DB = acton.database.ASCIIReader
    else:
        # Assume HDF5.
        is_pandas = bool(pandas_key)
        if is_pandas:
            logging.debug('Reading {} as pandas.'.format(data_path))
            DB = acton.database.PandasReader
            db_kwargs['key'] = pandas_key
        else:
            logging.debug('Reading {} as HDF5.'.format(data_path))
            DB = acton.database.HDF5Reader

    with DB(data_path, **db_kwargs) as reader:
        simulate_active_learning(reader.get_known_instance_ids(), reader,
                                 db_kwargs, output_path,
                                 n_epochs=n_epochs,
                                 n_initial_labels=initial_count,
                                 recommender=recommender,
                                 predictor=predictor,
                                 n_recommendations=n_recommendations)
