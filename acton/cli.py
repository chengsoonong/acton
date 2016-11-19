"""Command-line interface for Acton."""

import acton.acton
import click


@click.command()
@click.option('--data',
              type=click.Path(exists=True, dir_okay=False),
              help='Path to features/labels file',
              required=True)
@click.option('-f', '--feature',
              type=str,
              multiple=True,
              help='Column names of features',
              required=True)
@click.option('-l', '--label',
              type=str,
              help='Column name of labels',
              required=True)
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
              default='LogisticRegressionPredictor',
              help='Predictor to use')
@click.option('--recommender',
              type=click.Choice(acton.recommenders.RECOMMENDERS.keys()),
              default='QBCRecommender',
              help='Recommender to use')
@click.option('-o', '--output',
              type=click.Path(dir_okay=False),
              help='Path to output file',
              required=True)
def main(
        data: str,
        feature: str,
        label: str,
        epochs: int,
        id: str,
        diversity: float,
        recommendation_count: int,
        labeller_accuracy: float,
        predictor: str,
        recommender: str,
        output: str):
    acton.acton.main(data, feature, label, id)


if __name__ == '__main__':
    main()
