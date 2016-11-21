"""Command-line interface for Acton."""

import logging
import sys

import acton.acton
import acton.predictors
import acton.recommenders
import click


logging.root.setLevel(logging.DEBUG)


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
@click.option('--predictor',
              type=click.Choice(acton.predictors.PREDICTORS.keys()),
              default='LogisticRegression',
              help='Predictor to use')
@click.option('--recommender',
              type=click.Choice(acton.recommenders.RECOMMENDERS.keys()),
              default='RandomRecommender',
              help='Recommender to use')
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
        predictor: str,
        recommender: str,
):
    logging.warning('Not implemented: output, feature, diversity, '
                    'recommendation_count, labeller_accuracy, predictor, '
                    'recommender')
    return acton.acton.main(
        data_path=data,
        label_col=label,
        id_col=id,
        n_epochs=epochs)


if __name__ == '__main__':
    sys.exit(main())
