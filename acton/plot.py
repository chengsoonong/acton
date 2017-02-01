"""Script to plot a dump of predictions."""

import itertools
import sys
from typing import Iterable
from typing.io import BinaryIO

import acton.proto.io
from acton.proto.acton_pb2 import Predictions
import acton.proto.wrappers
import click
import matplotlib.pyplot as plt
import sklearn.metrics


def plot(predictions: Iterable[BinaryIO]):
    """Plots predictions from a file.

    Parameters
    ----------
    predictions
        Files containing predictions.
    """
    if len(predictions) < 1:
        raise ValueError('Must have at least 1 set of predictions.')

    metadata = []
    predictions, predictions_ = itertools.tee(predictions)
    for proto_file in predictions_:
        metadata.append(acton.proto.io.read_metadata(proto_file))

    for meta, proto_file in zip(metadata, predictions):
        # Read in the first protobuf to get the database file.
        protobuf = next(acton.proto.io.read_protos(proto_file, Predictions))
        protobuf = acton.proto.wrappers.Predictions(protobuf)
        with protobuf.DB() as db:
            accuracies = []
            for protobuf in acton.proto.io.read_protos(
                    proto_file, Predictions):
                protobuf = acton.proto.wrappers.Predictions(protobuf)
                ids = protobuf.predicted_ids
                predictions_ = protobuf.predictions
                assert predictions_.shape[0] == 1
                predictions_ = predictions_[0]
                labels = db.read_labels([0], ids).ravel()
                predicted_labels = predictions_.argmax(axis=1).ravel()
                accuracies.append(sklearn.metrics.accuracy_score(
                    labels, predicted_labels))

            plt.plot(accuracies, label=meta.decode('ascii', errors='replace'))

    plt.xlabel('Number of additional labels')
    plt.ylabel('Accuracy score')
    plt.legend()
    plt.show()


@click.command()
@click.argument('predictions',
                type=click.File('rb'),
                nargs=-1,
                required=True)
def _plot(predictions: Iterable[BinaryIO]):
    """Plots predictions from a file.

    Parameters
    ----------
    predictions
        Files containing predictions.
    """
    return plot(predictions)


if __name__ == '__main__':
    sys.exit(_plot())
