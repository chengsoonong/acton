#!/usr/bin/env python3

# This script simulates active learning using the component interface. The same
# task is accomplished in bash in simulate_active_learning.

from acton.acton import predict, recommend, label
import acton.database
from acton.proto.wrappers import Recommendations

# Initial labels.
recommendation_indices = list(range(10))
with acton.database.ASCIIReader(
        'tests/data/classification.txt',
        feature_cols=[], label_col='col20') as db:
    recommendations = Recommendations.make(
        recommended_ids=recommendation_indices,
        labelled_ids=[],
        recommender='None',
        db=db)
labels = label(recommendations)

# Main loop.
for epoch in range(10):
    print('Epoch', epoch)
    labels = label(
        recommend(predict(labels, 'LogisticRegression'), 'RandomRecommender'))

print('Labelled instances:', labels.ids)
