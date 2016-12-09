"""A predictor that uses KDE to classify instances."""

import numpy
import sklearn.base
import sklearn.neighbors
import sklearn.utils.multiclass
import sklearn.utils.validation


class KDEClassifier(sklearn.base.BaseEstimator, sklearn.base.ClassifierMixin):
    """A classifier using kernel density estimation to classify instances."""

    def __init__(self, bandwidth=1.0):
        self.bandwidth = bandwidth

    def fit(self, X, y):
        X, y = sklearn.utils.validation.check_X_y(X, y)

        self.classes_ = sklearn.utils.multiclass.unique_labels(y)

        self.kdes_ = [
            sklearn.neighbors.KernelDensity(self.bandwidth).fit(X[y == label])
            for label in self.classes_]

        return self

    def predict(self, X):
        sklearn.utils.validation.check_is_fitted(self, ['kdes_', 'classes_'])
        X = sklearn.utils.validation.check_array(X)

        scores = numpy.zeros((len(self.classes_), X.shape[0]))
        for label, kde in enumerate(self.kdes_):
            scores[label, :] = kde.score_samples(X)

        most_probable_indices = scores.argmax(axis=0)
        assert most_probable_indices.shape[0] == X.shape[0]

        return numpy.array([self.classes_[i] for i in most_probable_indices])
