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

        scores = self.predict_proba(X)

        most_probable_indices = scores.argmax(axis=1)
        assert most_probable_indices.shape[0] == X.shape[0]

        return numpy.array([self.classes_[i] for i in most_probable_indices])

    @staticmethod
    def _softmax(data, axis=0):
        """Computes the softmax of an array along an axis.

        Notes
        -----
        Adapted from https://gist.github.com/stober/1946926.

        Parameters
        ----------
        data : array_like
            Array of numbers.
        axis : int
            Axis to softmax along.
        """
        e_x = numpy.exp(
            data - numpy.expand_dims(numpy.max(data, axis=axis), axis))
        out = e_x / numpy.expand_dims(e_x.sum(axis=axis), axis)
        return out

    def predict_proba(self, X):
        """Predicts class probabilities.

        Class probabilities are normalised log densities of the kernel density
        estimates.

        Parameters
        ----------
        X : array_like, shape (n_samples, n_features)
            List of n_features-dimensional data points. Each row
            corresponds to a single data point.
        """
        sklearn.utils.validation.check_is_fitted(self, ['kdes_', 'classes_'])
        X = sklearn.utils.validation.check_array(X)

        scores = numpy.zeros((X.shape[0], len(self.classes_)))
        for label, kde in enumerate(self.kdes_):
            scores[:, label] = kde.score_samples(X)

        scores = self._softmax(scores, axis=1)

        assert scores.shape == (X.shape[0], len(self.classes_))
        assert numpy.allclose(scores.sum(axis=1), numpy.ones((X.shape[0],)))

        return scores
