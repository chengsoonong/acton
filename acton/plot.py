"""Script to plot a dump of predictions."""

import sys
from typing.io import BinaryIO

import acton.proto.io
from acton.proto.predictors_pb2 import Predictions
import acton.proto.wrappers
import click
import matplotlib.pyplot as plt
import sklearn.metrics


@click.command()
@click.argument('predictions',
                type=click.File('rb'),
                required=True)
def plot(predictions: BinaryIO):
    """Plots predictions from a file.

    Parameters
    ----------
    predictions
        Predictions file.
    """

    # Read in the first protobuf to get the database file.
    protobuf = next(acton.proto.io.read_protos(predictions, Predictions))
    protobuf = acton.proto.wrappers.PredictorOutput(protobuf)
    with protobuf.DB() as db:
        accuracies = []
        for protobuf in acton.proto.io.read_protos(predictions, Predictions):
            ids = protobuf.ids
            predictions = protobuf.predictions
            assert predictions.shape[0] == 1
            predictions = predictions[0]
            labels = db.read_labels(ids)
            predicted_labels = predictions.round()
            accuracies.append(sklearn.metrics.accuracy_score(
                labels, predicted_labels))

        plt.plot(accuracies)
        plt.xlabel('Number of additional labels')
        plt.ylabel('Accuracy score')
        plt.show()


if __name__ == '__main__':
    sys.exit(plot())
