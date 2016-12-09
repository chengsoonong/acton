"""A predictor that uses KDE to classify instances."""

import numpy
import sklearn.base
import sklearn.neighbors
import sklearn.utils.multiclass
import sklearn.utils.validation


class KDEClassifier(sklearn.base.BaseEstimator, sklearn.base.ClassifierMixin):
    """A classifier using kernel density estimation to classify instances."""

    def __init__(self, bandwidth=1.0):
        """A classifier using kernel density estimation to classify instances.

        A kernel density estimate is fit to each class. These estimates are used
        to score instances and the highest score class is used as the label for
        each instance.

        bandwidth : float
            Bandwidth for the kernel density estimate.
        """
        self.bandwidth = bandwidth

    def fit(self, X, y):
        """Fits kernel density models to the data.

        Parameters
        ----------
        X : array_like, shape (n_samples, n_features)
            List of n_features-dimensional data points. Each row
            corresponds to a single data point.
        y : array-like, shape (n_samples,)
            Target vector relative to X.
        """
        X, y = sklearn.utils.validation.check_X_y(X, y)

        self.classes_ = sklearn.utils.multiclass.unique_labels(y)

        self.kdes_ = [
            sklearn.neighbors.KernelDensity(self.bandwidth).fit(X[y == label])
            for label in self.classes_]

        return self

    def predict(self, X):
        """Predicts class labels.

        Parameters
        ----------
        X : array_like, shape (n_samples, n_features)
            List of n_features-dimensional data points. Each row
            corresponds to a single data point.
        """
        sklearn.utils.validation.check_is_fitted(self, ['kdes_', 'classes_'])
        X = sklearn.utils.validation.check_array(X)

        scores = numpy.zeros((len(self.classes_), X.shape[0]))
        for label, kde in enumerate(self.kdes_):
            scores[label, :] = kde.score_samples(X)

        most_probable_indices = scores.argmax(axis=0)
        assert most_probable_indices.shape[0] == X.shape[0]

        return numpy.array([self.classes_[i] for i in most_probable_indices])
