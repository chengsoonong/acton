"""Main processing script for Acton."""

import logging
import time
from typing import Iterable, List, TypeVar

import acton.database
import acton.labellers
import acton.predictors
import acton.proto.io
import acton.proto.wrappers
import acton.recommenders
import numpy
import pandas
import sklearn.linear_model
import sklearn.metrics
import sklearn.model_selection
import sklearn.preprocessing

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


def validate_predictor(predictor: str):
    """Raises an exception if the predictor is not valid.

    Parameters
    ----------
    predictor
        Name of predictor.

    Raises
    ------
    ValueError
    """
    if predictor not in acton.predictors.PREDICTORS:
        raise ValueError('Unknown predictor: {}. predictors are one of '
                         '{}.'.format(predictor,
                                      acton.predictors.PREDICTORS.keys()))


def validate_recommender(recommender: str):
    """Raises an exception if the recommender is not valid.

    Parameters
    ----------
    recommender
        Name of recommender.

    Raises
    ------
    ValueError
    """
    if recommender not in acton.recommenders.RECOMMENDERS:
        raise ValueError('Unknown recommender: {}. Recommenders are one of '
                         '{}.'.format(recommender,
                                      acton.recommenders.RECOMMENDERS.keys()))


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
    validate_recommender(recommender)
    validate_predictor(predictor)

    # Seed RNG.
    numpy.random.seed(0)

    # Bytestring describing this run.
    metadata = '{} | {}'.format(recommender, predictor).encode('ascii')

    # Split into training and testing sets.
    logging.debug('Found {} instances.'.format(len(ids)))
    logging.debug('Splitting into training/testing sets.')
    train_ids, test_ids = sklearn.model_selection.train_test_split(
        ids, test_size=test_size)
    test_ids.sort()

    # Set up predictor, labeller, and recommender.
    # TODO(MatthewJA): Handle multiple labellers better than just averaging.
    predictor_name = predictor  # For saving.
    predictor = acton.predictors.PREDICTORS[predictor](db=db, n_jobs=-1)

    labeller = acton.labellers.DatabaseLabeller(db)
    recommender = acton.recommenders.RECOMMENDERS[recommender](db=db)

    # Draw some initial labels.
    logging.debug('Drawing initial labels.')
    recommendations = draw(n_initial_labels, train_ids, replace=False)
    logging.debug('Recommending: {}'.format(recommendations))

    # This will store all IDs of things we have already labelled.
    labelled_ids = []
    # This will store all the corresponding labels.
    labels = numpy.zeros((0, 1))

    # Simulation loop.
    logging.debug('Writing protobufs to {}.'.format(output_path))
    writer = acton.proto.io.write_protos(output_path, metadata=metadata)
    next(writer)  # Prime the coroutine.
    for epoch in range(n_epochs):
        logging.info('Epoch {}/{}'.format(epoch + 1, n_epochs))
        # Label the recommendations.
        logging.debug('Labelling recommendations.')
        new_labels = numpy.array([
            labeller.query(id_) for id_ in recommendations]).reshape((-1, 1))

        labelled_ids.extend(recommendations)
        logging.debug('Sorting label IDs.')
        labelled_ids.sort()
        labels = numpy.concatenate([labels, new_labels], axis=0)

        # Here, we would write the labels to the database, but they're already
        # there since we're just reading them from there anyway.
        pass

        # Pass the labels to the predictor.
        logging.debug('Fitting predictor.')
        then = time.time()
        predictor.fit(labelled_ids)
        logging.debug('(Took {:.02} s.)'.format(time.time() - then))

        # Evaluate the predictor.
        logging.debug(
            'Making predictions (reference, n = {}).'.format(len(test_ids)))
        then = time.time()
        test_pred = predictor.reference_predict(test_ids)
        logging.debug('(Took {:.02} s.)'.format(time.time() - then))

        # Construct a protobuf for outputting predictions.
        proto = acton.proto.wrappers.Predictions.make(
            test_ids,
            labelled_ids,
            test_pred.transpose([1, 0, 2]),  # T x N x C -> N x T x C
            predictor=predictor_name,
            db=db)
        # Then write them to a file.
        logging.debug('Writing predictions.')
        writer.send(proto.proto)

        # Pass the predictions to the recommender.
        unlabelled_ids = list(set(ids) - set(labelled_ids))
        if not unlabelled_ids:
            logging.info('Labelled all instances.')
            break

        unlabelled_ids.sort()

        logging.debug(
            'Making predictions (unlabelled, n = {}).'.format(
                len(unlabelled_ids)))
        then = time.time()
        predictions = predictor.predict(unlabelled_ids)
        logging.debug('(Took {:.02} s.)'.format(time.time() - then))
        logging.debug('Making recommendations.')
        recommendations = recommender.recommend(
            unlabelled_ids, predictions, n=n_recommendations)
        logging.debug('Recommending: {}'.format(recommendations))

    return 0


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


def get_DB(
        data_path: str,
        pandas_key: str=None) -> (acton.database.Database, dict):
    """Gets a Database that will handle the given data table.

    Parameters
    ----------
    data_path
        Path to file.
    pandas_key
        Key for pandas HDF5. Specify iff using pandas.

    Returns
    -------
    Database
        Database that will handle the given data table.
    dict
        Keyword arguments for the Database constructor.
    """
    db_kwargs = {}

    is_fits = data_path.endswith('.fits')
    is_ascii = not data_path.endswith('.h5')
    if is_fits:
        logging.debug('Reading {} as FITS.'.format(data_path))
        DB = acton.database.FITSReader
    elif is_ascii:
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

    return DB, db_kwargs


def main(data_path: str, feature_cols: List[str], label_col: str,
         output_path: str, n_epochs: int=10, initial_count: int=10,
         recommender: str='RandomRecommender',
         predictor: str='LogisticRegression', pandas_key: str='',
         n_recommendations: int=1):
    """Simulate an active learning experiment.

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
    DB, db_kwargs = get_DB(data_path, pandas_key=pandas_key)

    db_kwargs['feature_cols'] = feature_cols
    db_kwargs['label_col'] = label_col

    with DB(data_path, **db_kwargs) as reader:
        return simulate_active_learning(reader.get_known_instance_ids(), reader,
                                        db_kwargs, output_path,
                                        n_epochs=n_epochs,
                                        n_initial_labels=initial_count,
                                        recommender=recommender,
                                        predictor=predictor,
                                        n_recommendations=n_recommendations)


def predict(
        labels: acton.proto.wrappers.LabelPool,
        predictor: str) -> acton.proto.wrappers.Predictions:
    """Train a predictor and predict labels.

    Parameters
    ---------
    labels
        IDs of labelled instances.
    predictor
        Name of predictor to make predictions.
    """
    validate_predictor(predictor)

    with labels.DB() as db:
        ids = db.get_known_instance_ids()
        train_ids = labels.ids

        predictor_name = predictor
        predictor = acton.predictors.PREDICTORS[predictor](db=db, n_jobs=-1)

        logging.debug('Training predictor with IDs: {}'.format(train_ids))
        predictor.fit(train_ids)

        predictions = predictor.reference_predict(ids)

        # Construct a protobuf for outputting predictions.
        proto = acton.proto.wrappers.Predictions.make(
            ids,
            train_ids,
            predictions.transpose([1, 0, 2]),  # T x N x C -> N x T x C
            predictor=predictor_name,
            db=db)
        return proto


def recommend(
        predictions: acton.proto.wrappers.Predictions,
        recommender: str='RandomRecommender',
        n_recommendations: int=1) -> acton.proto.wrappers.Recommendations:
    """Recommends instances to label based on predictions.

    Parameters
    ---------
    recommender
        Name of recommender to make recommendations.
    n_recommendations
        Number of recommendations to make at once. Default 1.

    Returns
    -------
    acton.proto.wrappers.Recommendations
    """
    validate_recommender(recommender)

    # Make a list of IDs that do not have labels and the indices of the
    # corresponding predictions.
    ids = []
    indices = []
    has_labels = set(predictions.labelled_ids)
    for pred_index, id_ in enumerate(predictions.predicted_ids):
        if id_ not in has_labels:
            ids.append(id_)
            indices.append(pred_index)
    # Array of predictions for unlabelled instances.
    predictions_array = predictions.predictions[:, indices]

    with predictions.DB() as db:
        recommender_name = recommender
        recommender = acton.recommenders.RECOMMENDERS[recommender](db=db)
        recommendations = recommender.recommend(
            ids, predictions_array, n=n_recommendations)

        logging.debug('Recommending: {}'.format(list(recommendations)))

        # Construct a protobuf for outputting recommendations.
        proto = acton.proto.wrappers.Recommendations.make(
            [int(r) for r in recommendations],
            predictions.labelled_ids,
            recommender=recommender_name,
            db=db)
        return proto


def label(recommendations: acton.proto.wrappers.Recommendations
          ) -> acton.proto.wrappers.LabelPool:
    """Simulates a labelling task.

    Parameters
    ---------
    data_path
        Path to data file.
    feature_cols
        List of column names of features. If empty, all columns will be used.
    label_col
        Column name of the labels.
    pandas_key
        Key for pandas HDF5. Specify iff using pandas.

    Returns
    -------
    acton.proto.wrappers.LabelPool
    """
    # We'd store the labels here, except that we just read them from the DB.
    # Instead, we'll record that we've labelled them.
    # # labeller = acton.labellers.DatabaseLabeller(db)
    # # labels = [labeller.query(id_) for id_ in ids]

    # TODO(MatthewJA): Consider optimising this (doesn't really need a sort).
    ids_to_label = recommendations.recommendations
    labelled_ids = recommendations.labelled_ids
    logging.debug('Recommended IDs: {}'.format(ids_to_label))
    logging.debug('Already labelled IDs: {}'.format(labelled_ids))
    ids = sorted(set(ids_to_label) | set(labelled_ids))
    logging.debug('Now labelled IDs: {}'.format(ids))

    # Return a protobuf.
    with recommendations.DB() as db:
        proto = acton.proto.wrappers.LabelPool.make(ids=ids, db=db)
    return proto
