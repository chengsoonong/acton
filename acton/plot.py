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
                nargs=-1,
                required=True)
def plot(predictions: BinaryIO):
    """Plots predictions from a file.

    Parameters
    ----------
    predictions
        Predictions file.
    """

    if len(predictions) < 1:
        raise ValueError('Must have at least 1 set of predictions.')

    for predictions_ in predictions:
        # Read in the first protobuf to get the database file.
        protobuf = next(acton.proto.io.read_protos(predictions_, Predictions))
        protobuf = acton.proto.wrappers.PredictorOutput(protobuf)
        with protobuf.DB() as db:
            accuracies = []
            for protobuf in acton.proto.io.read_protos(
                    predictions_, Predictions):
                protobuf = acton.proto.wrappers.PredictorOutput(protobuf)
                ids = protobuf.ids
                predictions_ = protobuf.predictions
                assert predictions_.shape[0] == 1
                predictions_ = predictions_[0]
                labels = db.read_labels([b'0'], ids).ravel()
                predicted_labels = predictions_.round().ravel()
                accuracies.append(sklearn.metrics.accuracy_score(
                    labels, predicted_labels))

            plt.plot(accuracies, label=protobuf.proto.predictor)

    plt.xlabel('Number of additional labels')
    plt.ylabel('Accuracy score')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    sys.exit(plot())
