"""Command-line interface for Acton."""

import logging
import struct
import sys
from typing import BinaryIO, Iterable, List

import acton.acton
import acton.predictors
import acton.proto.wrappers
import acton.recommenders
import click


def read_bytes_from_buffer(n: int, buffer: BinaryIO) -> bytes:
    """Reads n bytes from stdin, blocking until all bytes are received.

    Parameters
    ----------
    n
        How many bytes to read.
    buffer
        Which buffer to read from.

    Returns
    -------
    bytes
        Exactly n bytes.
    """
    b = b''
    while len(b) < n:
        b += buffer.read(n - len(b))
    assert len(b) == n
    return b


def read_binary() -> bytes:
    """Reads binary data from stdin.

    Notes
    -----
    The first eight bytes are expected to be the length of the input data as an
    unsigned long long.

    Returns
    -------
    bytes
        Binary data.
    """
    logging.debug('Reading 8 bytes from stdin.')
    length = read_bytes_from_buffer(8, sys.stdin.buffer)
    length, = struct.unpack('<Q', length)
    logging.debug('Reading {} bytes from stdin.'.format(length))
    return read_bytes_from_buffer(length, sys.stdin.buffer)


def write_binary(string: bytes):
    """Writes binary data to stdout.

    Notes
    -----
    The output will be preceded by the length as an unsigned long long.
    """
    logging.debug('Writing 8 + {} bytes to stdout.'.format(len(string)))
    length = struct.pack('<Q', len(string))
    logging.debug('Writing length {} ({}).'.format(length, len(string)))
    sys.stdout.buffer.write(length)
    sys.stdout.buffer.write(string)
    sys.stdout.buffer.flush()


# acton


@click.command()
@click.option('--data',
              type=click.Path(exists=True, dir_okay=False),
              help='Path to features/labels file',
              required=True)
@click.option('-l', '--label',
              type=str,
              help='Column name of labels',
              required=True)
@click.option('-o', '--output',
              type=click.Path(dir_okay=False),
              help='Path to output file',
              required=True)
@click.option('-f', '--feature',
              type=str,
              multiple=True,
              help='Column names of features')
@click.option('--epochs',
              type=int,
              help='Number of epochs to run active learning for',
              default=100)
@click.option('-i', '--id',
              type=str,
              help='Column name of IDs')
@click.option('--diversity',
              type=float,
              help='Diversity of recommendations',
              default=0.0)
@click.option('--recommendation-count',
              type=int,
              help='Number of recommendations to make',
              default=1)
@click.option('--labeller-accuracy',
              type=float,
              help='Accuracy of simulated labellers',
              default=1.0)
@click.option('--initial-count',
              type=int,
              help='Number of random instances to label initially',
              default=10)
@click.option('--predictor',
              type=click.Choice(acton.predictors.PREDICTORS.keys()),
              default='LogisticRegression',
              help='Predictor to use')
@click.option('--recommender',
              type=click.Choice(acton.recommenders.RECOMMENDERS.keys()),
              default='RandomRecommender',
              help='Recommender to use')
@click.option('--pandas-key',
              type=str,
              default='',
              help='Key for pandas HDF5')
@click.option('-v', '--verbose',
              is_flag=True,
              help='Verbose output')
def main(
        data: str,
        label: str,
        output: str,
        feature: str,
        epochs: int,
        id: str,
        diversity: float,
        recommendation_count: int,
        labeller_accuracy: float,
        initial_count: int,
        predictor: str,
        recommender: str,
        verbose: bool,
        pandas_key: str,
):
    logging.warning('Not implemented: diversity, id_col, labeller_accuracy')
    logging.captureWarnings(True)
    if verbose:
        logging.root.setLevel(logging.DEBUG)
    return acton.acton.main(
        data_path=data,
        feature_cols=feature,
        label_col=label,
        output_path=output,
        n_epochs=epochs,
        initial_count=initial_count,
        recommender=recommender,
        predictor=predictor,
        pandas_key=pandas_key,
        n_recommendations=recommendation_count)


# acton-predict


@click.command()
@click.option('--predictor',
              type=click.Choice(acton.predictors.PREDICTORS.keys()),
              default='LogisticRegression',
              help='Predictor to use')
@click.option('-v', '--verbose',
              is_flag=True,
              help='Verbose output')
def predict(
        predictor: str,
        verbose: bool,
):
    # Logging setup.
    logging.captureWarnings(True)
    if verbose:
        logging.root.setLevel(logging.DEBUG)

    # Read labels.
    labels = read_binary()
    labels = acton.proto.wrappers.LabelPool.deserialise(labels)

    # Write predictions.
    proto = acton.acton.predict(labels=labels, predictor=predictor)
    write_binary(proto.proto.SerializeToString())


# acton-recommend


@click.command()
@click.option('--diversity',
              type=float,
              help='Diversity of recommendations',
              default=0.0)
@click.option('--recommendation-count',
              type=int,
              help='Number of recommendations to make',
              default=1)
@click.option('--recommender',
              type=click.Choice(acton.recommenders.RECOMMENDERS.keys()),
              default='RandomRecommender',
              help='Recommender to use')
@click.option('-v', '--verbose',
              is_flag=True,
              help='Verbose output')
def recommend(
        diversity: float,
        recommendation_count: int,
        recommender: str,
        verbose: bool,
):
    # Logging setup.
    logging.warning('Not implemented: diversity')
    logging.captureWarnings(True)
    if verbose:
        logging.root.setLevel(logging.DEBUG)

    # Read the predictions protobuf.
    predictions = read_binary()
    predictions = acton.proto.wrappers.Predictions.deserialise(predictions)

    # Write the recommendations protobuf.
    proto = acton.acton.recommend(
        predictions=predictions,
        recommender=recommender,
        n_recommendations=recommendation_count)
    write_binary(proto.proto.SerializeToString())


# acton-label


def lines_from_stdin() -> Iterable[str]:
    """Yields lines from stdin."""
    for line in sys.stdin:
        line = line.strip()
        logging.debug('Read line {} from stdin.'.format(repr(line)))
        if line:
            yield line


@click.command()
@click.option('--data',
              type=click.Path(exists=True, dir_okay=False),
              help='Path to labels file',
              required=False)
@click.option('-l', '--label',
              type=str,
              help='Column name of labels',
              required=False)
@click.option('-f', '--feature',
              type=str,
              multiple=True,
              help='Column names of features')
@click.option('--labeller-accuracy',
              type=float,
              help='Accuracy of simulated labellers',
              default=1.0)
@click.option('--pandas-key',
              type=str,
              default='',
              help='Key for pandas HDF5')
@click.option('-v', '--verbose',
              is_flag=True,
              help='Verbose output')
def label(
        data: str,
        feature: List[str],
        label: str,
        labeller_accuracy: float,
        verbose: bool,
        pandas_key: str,
):
    # Logging setup.
    logging.warning('Not implemented: labeller_accuracy')
    logging.captureWarnings(True)
    if verbose:
        logging.root.setLevel(logging.DEBUG)

    # If any arguments are specified, expect all arguments.
    if data or label or pandas_key:
        if not data or not label:
            raise ValueError('--data, --label, or --pandas-key specified, but '
                             'missing --data or --label.')

        # Handle database arguments.
        data_path = data
        feature_cols = feature
        label_col = label

        # Read IDs from stdin.
        ids_to_label = [int(i) for i in lines_from_stdin()]

        # There wasn't a recommendations protobuf given, so we have no existing
        # labelled instances.
        labelled_ids = []

        # Construct the recommendations protobuf.
        DB, db_kwargs = acton.acton.get_DB(data_path, pandas_key=pandas_key)
        db_kwargs['label_col'] = label_col
        db_kwargs['feature_cols'] = feature_cols
        with DB(data_path, **db_kwargs) as db:
            recs = acton.proto.wrappers.Recommendations.make(
                recommended_ids=ids_to_label,
                labelled_ids=labelled_ids,
                recommender='None',
                db=db)
    else:
        # Read a recommendations protobuf from stdin.
        recs = read_binary()
        recs = acton.proto.wrappers.Recommendations.deserialise(recs)

    proto = acton.acton.label(recs)
    write_binary(proto.proto.SerializeToString())


if __name__ == '__main__':
    sys.exit(main())
