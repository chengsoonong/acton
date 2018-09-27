"""Predictor classes."""

from abc import ABC, abstractmethod
import logging
from typing import Iterable, Sequence

import acton.database
import acton.kde_predictor
import GPy as gpy
import numpy
import sklearn.base
import sklearn.linear_model
import sklearn.model_selection
import sklearn.preprocessing
from numpy.random import multivariate_normal, gamma, multinomial



class Predictor(ABC):
    """Base class for predictors.

    Attributes
    ----------
    prediction_type : str
        What kind of predictions this class generates, e.g. classification.s
    """
    prediction_type = 'classification'

    @abstractmethod
    def fit(self, ids: Iterable[int]):
        """Fits the predictor to labelled data.

        Parameters
        ----------
        ids
            List of IDs of instances to train from.
        """

    @abstractmethod
    def predict(self, ids: Sequence[int]) -> (numpy.ndarray, numpy.ndarray):
        """Predicts labels of instances.

        Notes
        -----
            Unlike in scikit-learn, predictions are always real-valued.
            Predicted labels for a classification problem are represented by
            predicted probabilities of each class.

        Parameters
        ----------
        ids
            List of IDs of instances to predict labels for.

        Returns
        -------
        numpy.ndarray
            An N x T x C array of corresponding predictions.
        numpy.ndarray
            A N array of confidences (or None if not applicable).
        """

    @abstractmethod
    def reference_predict(
            self, ids: Sequence[int]) -> (numpy.ndarray, numpy.ndarray):
        """Predicts labels using the best possible method.

        Parameters
        ----------
        ids
            List of IDs of instances to predict labels for.

        Returns
        -------
        numpy.ndarray
            An N x 1 x C array of corresponding predictions.
        numpy.ndarray
            A N array of confidences (or None if not applicable).
        """


class _InstancePredictor(Predictor):
    """Wrapper for a scikit-learn instance.

    Attributes
    ----------
    _db : acton.database.Database
        Database storing features and labels.
    _instance : sklearn.base.BaseEstimator
        scikit-learn predictor instance.
    """

    def __init__(self, instance: sklearn.base.BaseEstimator,
                 db: acton.database.Database):
        """
        Arguments
        ---------
        instance
            scikit-learn predictor instance.
        db
            Database storing features and labels.
        """
        self._db = db
        self._instance = instance

    def fit(self, ids: Iterable[int]):
        """Fits the predictor to labelled data.

        Parameters
        ----------
        ids
            List of IDs of instances to train from.
        """
        features = self._db.read_features(ids)
        labels = self._db.read_labels([0], ids)
        self._instance.fit(features, labels.ravel())

    def predict(self, ids: Sequence[int]) -> (numpy.ndarray, None):
        """Predicts labels of instances.

        Notes
        -----
            Unlike in scikit-learn, predictions are always real-valued.
            Predicted labels for a classification problem are represented by
            predicted probabilities of each class.

        Parameters
        ----------
        ids
            List of IDs of instances to predict labels for.

        Returns
        -------
        numpy.ndarray
            An N x 1 x C array of corresponding predictions.
        numpy.ndarray
            A N array of confidences (or None if not applicable).
        """
        features = self._db.read_features(ids)
        try:
            probs = self._instance.predict_proba(features)
            return probs.reshape((probs.shape[0], 1, probs.shape[1])), None
        except AttributeError:
            probs = self._instance.predict(features)
            if len(probs.shape) == 1:
                return probs.reshape((probs.shape[0], 1, 1)), None
            else:
                raise NotImplementedError()

    def reference_predict(self, ids: Sequence[int]) -> (numpy.ndarray, None):
        """Predicts labels using the best possible method.

        Parameters
        ----------
        ids
            List of IDs of instances to predict labels for.

        Returns
        -------
        numpy.ndarray
            An N x 1 x C array of corresponding predictions.
        numpy.ndarray
            A N array of confidences (or None if not applicable).
        """
        return self.predict(ids)


def from_instance(predictor: sklearn.base.BaseEstimator,
                  db: acton.database.Database, regression: bool=False
                  ) -> Predictor:
    """Converts a scikit-learn predictor instance into a Predictor instance.

    Arguments
    ---------
    predictor
        scikit-learn predictor.
    db
        Database storing features and labels.
    regression
        Whether this predictor does regression (as opposed to classification).

    Returns
    -------
    Predictor
        Predictor instance wrapping the scikit-learn predictor.
    """
    ip = _InstancePredictor(predictor, db)
    if regression:
        ip.prediction_type = 'regression'
    return ip


def from_class(Predictor: type, regression: bool=False) -> type:
    """Converts a scikit-learn predictor class into a Predictor class.

    Arguments
    ---------
    Predictor
        scikit-learn predictor class.
    regression
        Whether this predictor does regression (as opposed to classification).

    Returns
    -------
    type
        Predictor class wrapping the scikit-learn class.
    """
    class Predictor_(_InstancePredictor):

        def __init__(self, db, **kwargs):
            super().__init__(instance=None, db=db)
            self._instance = Predictor(**kwargs)

    if regression:
        Predictor_.prediction_type = 'regression'

    return Predictor_


class Committee(Predictor):
    """A predictor using a committee of other predictors.

    Attributes
    ----------
    n_classifiers : int
        Number of logistic regression classifiers in the committee.
    subset_size : float
        Percentage of known labels to take subsets of to train the
        classifier. Lower numbers increase variety.
    _db : acton.database.Database
        Database storing features and labels.
    _committee : List[sklearn.linear_model.LogisticRegression]
        Underlying committee of logistic regression classifiers.
    _reference_predictor : Predictor
        Reference predictor trained on all known labels.
    """

    def __init__(self, Predictor: type, db: acton.database.Database,
                 n_classifiers: int=10, subset_size: float=0.6,
                 **kwargs: dict):
        """
        Parameters
        ----------
        Predictor
            Predictor to use in the committee.
        db
            Database storing features and labels.
        n_classifiers
            Number of logistic regression classifiers in the committee.
        subset_size
            Percentage of known labels to take subsets of to train the
            classifier. Lower numbers increase variety.
        kwargs
            Keyword arguments passed to the underlying Predictor.
        """
        self.n_classifiers = n_classifiers
        self.subset_size = subset_size
        self._db = db
        self._committee = [Predictor(db=db, **kwargs)
                           for _ in range(n_classifiers)]
        self._reference_predictor = Predictor(db=db, **kwargs)

    def fit(self, ids: Iterable[int]):
        """Fits the predictor to labelled data.

        Parameters
        ----------
        ids
            List of IDs of instances to train from.
        """
        # Get labels so we can stratify a split.
        labels = self._db.read_labels([0], ids)
        for classifier in self._committee:
            # Take a subsets to introduce variety.
            try:
                subset, _ = sklearn.model_selection.train_test_split(
                    ids, train_size=self.subset_size, stratify=labels)
            except ValueError:
                # Too few labels.
                subset = ids
            classifier.fit(subset)
        self._reference_predictor.fit(ids)

    def predict(self, ids: Sequence[int]) -> (numpy.ndarray, numpy.ndarray):
        """Predicts labels of instances.

        Notes
        -----
            Unlike in scikit-learn, predictions are always real-valued.
            Predicted labels for a classification problem are represented by
            predicted probabilities of each class.

        Parameters
        ----------
        ids
            List of IDs of instances to predict labels for.

        Returns
        -------
        numpy.ndarray
            An N x T x C array of corresponding predictions.
        numpy.ndarray
            A N array of confidences (or None if not applicable).
        """
        predictions = numpy.concatenate(
            [classifier.predict(ids)[0]
             for classifier in self._committee],
            axis=1)
        assert predictions.shape[:2] == (len(ids), len(self._committee))
        stdevs = predictions.std(axis=1).mean(axis=1)
        return predictions, stdevs

    def reference_predict(
            self, ids: Sequence[int]) -> (numpy.ndarray, numpy.ndarray):
        """Predicts labels using the best possible method.

        Parameters
        ----------
        ids
            List of IDs of instances to predict labels for.

        Returns
        -------
        numpy.ndarray
            An N x 1 x C array of corresponding predictions.
        numpy.ndarray
            A N array of confidences (or None if not applicable).
        """
        _, stdevs = self.predict(ids)
        return self._reference_predictor.predict(ids)[0], stdevs


def AveragePredictions(predictor: Predictor) -> Predictor:
    """Wrapper for a predictor that averages predicted probabilities.

    Notes
    -----
    This effectively reduces the number of predictors to 1.

    Arguments
    ---------
    predictor
        Predictor to wrap.

    Returns
    -------
    Predictor
        Predictor with averaged predictions.
    """
    predictor.predict_ = predictor.predict

    def predict(features: numpy.ndarray) -> (numpy.ndarray, numpy.ndarray):
        predictions, stdevs = predictor.predict_(features)
        predictions = predictions.mean(axis=1)
        return predictions.reshape(
            (predictions.shape[0], 1, predictions.shape[1])), stdevs

    predictor.predict = predict

    return predictor


class GPClassifier(Predictor):
    """Classifier using Gaussian processes.

    Attributes
    ----------
    max_iters : int
        Maximum optimisation iterations.
    label_encoder : sklearn.preprocessing.LabelEncoder
        Encodes labels as integers.
    model_ : gpy.models.GPClassification
        GP model.
    _db : acton.database.Database
        Database storing features and labels.
    """
    def __init__(self, db: acton.database.Database, max_iters: int=50000,
                 n_jobs: int=1):
        """
        Parameters
        ----------
        db
            Database.
        max_iters
            Maximum optimisation iterations.
        n_jobs
            Does nothing; here for compatibility with sklearn.
        """
        self._db = db
        self.max_iters = max_iters

    def fit(self, ids: Iterable[int]):
        """Fits the predictor to labelled data.

        Parameters
        ----------
        ids
            List of IDs of instances to train from.
        """
        features = self._db.read_features(ids)
        labels = self._db.read_labels([0], ids).ravel()
        self.label_encoder_ = sklearn.preprocessing.LabelEncoder()
        labels = self.label_encoder_.fit_transform(labels).reshape((-1, 1))
        if len(self.label_encoder_.classes_) > 2:
            raise ValueError(
                'GPClassifier only supports binary classification.')
        self.model_ = gpy.models.GPClassification(features, labels)
        self.model_.optimize('bfgs', max_iters=self.max_iters)

    def predict(self, ids: Sequence[int]) -> (numpy.ndarray, numpy.ndarray):
        """Predicts labels of instances.

        Notes
        -----
            Unlike in scikit-learn, predictions are always real-valued.
            Predicted labels for a classification problem are represented by
            predicted probabilities of each class.

        Parameters
        ----------
        ids
            List of IDs of instances to predict labels for.

        Returns
        -------
        numpy.ndarray
            An N x 1 x C array of corresponding predictions.
        numpy.ndarray
            A N array of confidences (or None if not applicable).
        """
        features = self._db.read_features(ids)
        p_predictions, variances = self.model_.predict(features)
        n_predictions = 1 - p_predictions
        predictions = numpy.concatenate([n_predictions, p_predictions], axis=1)

        logging.debug('Variance: {}'.format(variances))
        if isinstance(variances, float) and numpy.isnan(variances):
            variances = None
        else:
            variances = variances.ravel()
            assert variances.shape == (len(ids),)
        assert predictions.shape[1] == 2
        return predictions.reshape((-1, 1, 2)), variances

    def reference_predict(
            self, ids: Sequence[int]) -> (numpy.ndarray, numpy.ndarray):
        """Predicts labels using the best possible method.

        Parameters
        ----------
        ids
            List of IDs of instances to predict labels for.

        Returns
        -------
        numpy.ndarray
            An N x 1 x C array of corresponding predictions.
        numpy.ndarray
            A N array of confidences (or None if not applicable).
        """
        return self.predict(ids)

class TensorPredictor(Predictor):
    """Predict labels for each tensor entry.

    Attributes
    ----------
    _db : acton.database.Database
        Database storing features and labels.
    n_particles:
        Number of particles for Thompson sampling.
    n_relations:
        Number of relations (K)
    n_entities:
        Number of entities (N)
    n_dim
        Number of latent dimensions (D)
    _var_r
        variance of prior of R
    _var_e 
        variance of prior of E 
    var_x
        variance of X
    sample_prior
        indicates whether sample prior
    E
        P x N x D entity features
    R
        P x K x D x D relation features
    X 
        K x N x N labels
    """

    def __init__(self,
                 db: acton.database.Database,
                 n_particles : int = 5,
                 _var_r : int = 1, _var_e: int = 1,
                 var_x : float = 0.1,
                 sample_prior : bool = False,
                 n_jobs: int=1):
        """
        Arguments
        ---------
        db
            Database storing features and labels.
        n_particles:
            Number of particles for Thompson sampling.
        _var_r
            variance of prior of R
        _var_e 
            variance of prior of E 
        var_x
            variance of X
        sample_prior
            indicates whether sample prior
        """
        self._db = db
        self.n_particles = n_particles
        self._var_r = _var_r
        self._var_e = _var_e
        self.var_x = var_x

        self.var_e = numpy.ones(self.n_particles) * self._var_e
        self.var_r = numpy.ones(self.n_particles) * self._var_r

        self.p_weights = numpy.ones(self.n_particles) / self.n_particles

        self.sample_prior = sample_prior

        self.E, self.R = self._db.read_features()
        #X : numpy.ndarray
        #    Fully observed tensor with shape (n_relations, n_entities, n_entities)
        all_ = []
        self.X = self._db.read_labels(all_) # read all labels


    def fit(self, ids: Iterable[tuple]):
        """Update posteriors.

        Parameters
        ----------
        ids
            List of IDs of labelled instances.

        Returns
        -------
        seq : (numpy.ndarray, numpy.ndarray)
            Returns a updated posteriors for E and R.
        """

        assert self.n_particles == self.E.shape[0] == self.R.shape[0]
        self.n_relations = self.X.shape[0]
        self.n_entities = self.X.shape[1]
        self.n_dim = self.E.shape[2]
        assert self.E.shape[2] == self.R.shape[2]

        obs_mask = numpy.zeros_like(self.X)

        for _id in ids:
            r_k, e_i, e_j = _id
            obs_mask[r_k, e_i, e_j] = 1

        cur_obs = numpy.zeros_like(self.X)
        for k in range(self.n_relations):
            cur_obs[k][obs_mask[k] == 1] = self.X[k][obs_mask[k] == 1]

        # cur_obs[cur_obs.nonzero()] = 1
        self.obs_sum = numpy.sum(numpy.sum(obs_mask, 1), 1)
        self.valid_relations = numpy.nonzero(numpy.sum(numpy.sum(self.X, 1), 1))[0]

        self.features = numpy.zeros([2 * self.n_entities * self.n_relations, self.n_dim])
        self.xi = numpy.zeros([2 * self.n_entities * self.n_relations])

        # only consider the situation where one element is recommended each time
        next_idx = ids[-1]

        self.p_weights *= self.compute_particle_weight(next_idx, cur_obs, obs_mask)
        self.p_weights /= numpy.sum(self.p_weights)


        ESS = 1. / numpy.sum((self.p_weights ** 2))

        if ESS < self.n_particles / 2.:
            self.resample()

        for p in range(self.n_particles):
            #time_before_sample_relations = time.time()
            self._sample_relations(cur_obs, obs_mask, self.E[p], self.R[p], self.var_r[p])
            #time_after_sample_relations = time.time()
            #logging.debug('Sample all relations took: {} s'.format(time_after_sample_relations - time_before_sample_relations))
            self._sample_entities(cur_obs, obs_mask, self.E[p], self.R[p], self.var_e[p])
            #time_after_sample_entities = time.time()
            #logging.debug('Sample all entities took: {} s'.format(time_after_sample_entities - time_after_sample_relations))

        if self.sample_prior and i != 0 and i % self.prior_sample_gap == 0:
            self._sample_prior()

    def predict(self, ids: Sequence[int] = None) -> (numpy.ndarray, None):
        """Predicts labels of instances.

        Notess
        -----
            Unlike in scikit-learn, predictions are always real-valued.
            Predicted labels for a classification problem are represented by
            predicted probabilities of each class.

        Parameters
        ----------
        ids
            List of IDs of instances to predict labels for.

        Returns
        -------
        numpy.ndarray
            An K x D x D  array of corresponding predictions.
        """
        p = multinomial(1, self.p_weights).argmax()

        # reconstruct
        X = numpy.zeros([self.n_relations, self.n_entities, self.n_entities])

        for k in range(self.n_relations):
            X[k] = numpy.dot(numpy.dot(self.E[p], self.R[p][k]), self.E[p].T)

        #logging.critical('R[0, 2,4]: {}'.format(self.R[0,2,4]))

        return X, None


    def reference_predict(self, ids: Sequence[int]) -> (numpy.ndarray, None):
        """Predicts labels using the best possible method.

        Parameters
        ----------
        ids
            List of IDs of instances to predict labels for.

        Returns
        -------
        numpy.ndarray
            An N x 1 x C array of corresponding predictions.
        numpy.ndarray
            A N array of confidences (or None if not applicable).
        """
        return self.predict(ids)


    def _sample_prior(self):
        self._sample_var_r()
        self._sample_var_e()

    def resample(self):
        count = multinomial(self.n_particles, self.p_weights)

        logging.debug("[RESAMPLE] %s", str(count))

        new_E = list()
        new_R = list()

        for p in range(self.n_particles):
            for i in range(count[p]):
                new_E.append(self.E[p].copy())
                new_R.append(self.R[p].copy())

        self.E = numpy.asarray(new_E)
        self.R = numpy.asarray(new_R)
        self.p_weights = numpy.ones(self.n_particles) / self.n_particles

    def compute_particle_weight(self, next_idx, X, mask):
        from scipy.stats import norm
        r_k, e_i, e_j = next_idx

        log_weight = numpy.zeros(self.n_particles)
        for p in range(self.n_particles):

            mean = numpy.dot(numpy.dot(self.E[p][e_i], self.R[p][r_k]), self.E[p][e_j])
            log_weight[p] = norm.logpdf(X[next_idx], mean, self.var_x)

        log_weight -= numpy.max(log_weight)
        weight = numpy.exp(log_weight)
        weight += 1e-10
        return weight / numpy.sum(weight)

    def _sample_var_r(self):
        for p in range(self.n_particles):
            self.var_r[p] = 1. / gamma(0.5 * self.n_relations * self.n_dim * self.n_dim + self.r_alpha,
                                       1. / (0.5 * numpy.sum(self.R[p] ** 2) + self.r_beta))
        logging.debug("Sampled var_r %.3f", numpy.mean(self.var_r))

    def _sample_var_e(self):
        for p in range(self.n_particles):
            self.var_e[p] = 1. / gamma(0.5 * self.n_entities * self.n_dim + self.e_alpha,
                                       1. / (0.5 * numpy.sum(self.E[p] ** 2) + self.e_beta))
        logging.debug("Sampled var_e %.3f", numpy.mean(self.var_e))

    def _sample_entities(self, X, mask, E, R, var_e, sample_idx=None):
        RE = numpy.zeros([self.n_relations, self.n_entities, self.n_dim])
        RTE = numpy.zeros([self.n_relations, self.n_entities, self.n_dim])

        for k in range(self.n_relations):
            RE[k] = numpy.dot(R[k], E.T).T
            RTE[k] = numpy.dot(R[k].T, E.T).T

        if isinstance(sample_idx, type(None)):
            sample_idx = range(self.n_entities)

        for i in sample_idx:
            self._sample_entity(X, mask, E, R, i, var_e, RE, RTE)
            for k in range(self.n_relations):
                RE[k][i] = numpy.dot(R[k], E[i])
                RTE[k][i] = numpy.dot(R[k].T, E[i])

    def _sample_entity(self, X, mask, E, R, i, var_e, RE=None, RTE=None):
        nz_r = mask[:, i, :].nonzero()
        nz_c = mask[:, :, i].nonzero()
        nnz_r = nz_r[0].size
        nnz_c = nz_c[0].size
        nnz_all = nnz_r + nnz_c

        self.features[:nnz_r] = RE[nz_r]
        self.features[nnz_r:nnz_all] = RTE[nz_c]
        self.xi[:nnz_r] = X[:, i, :][nz_r]
        self.xi[nnz_r:nnz_all] = X[:, :, i][nz_c]
        _xi = self.xi[:nnz_all] * self.features[:nnz_all].T
        xi = numpy.sum(_xi, 1) / self.var_x

        _lambda = numpy.identity(self.n_dim) / var_e
        _lambda += numpy.dot(self.features[:nnz_all].T, self.features[:nnz_all]) / self.var_x

        # mu = numpy.linalg.solve(_lambda, xi)
        # E[i] = normal(mu, _lambda)

        inv_lambda = numpy.linalg.inv(_lambda)
        mu = numpy.dot(inv_lambda, xi)
        E[i] = multivariate_normal(mu, inv_lambda)

        numpy.mean(numpy.diag(inv_lambda))
        # logging.info('Mean variance E, %d, %f', i, mean_var)


    def _sample_relations(self, X, mask, E, R, var_r):
        EXE = numpy.kron(E, E)

        for k in self.valid_relations:
            if self.obs_sum[k] != 0:
                self._sample_relation(X, mask, E, R, k, EXE, var_r)
            else:
                R[k] = numpy.random.normal(0, var_r, size=[self.n_dim, self.n_dim])

    def _sample_relation(self, X, mask, E, R, k, EXE, var_r):
        _lambda = numpy.identity(self.n_dim ** 2) / var_r
        xi = numpy.zeros(self.n_dim ** 2)

        kron = EXE[mask[k].flatten() == 1]

        if kron.shape[0] != 0:
            _lambda += numpy.dot(kron.T, kron)
            xi += numpy.sum(X[k, mask[k] == 1].flatten() * kron.T, 1)

        _lambda /= self.var_x
        # mu = numpy.linalg.solve(_lambda, xi) / self.var_x

        inv_lambda = numpy.linalg.inv(_lambda)
        mu = numpy.dot(inv_lambda, xi) / self.var_x
        try:
            # R[k] = normal(mu, _lambda).reshape([self.n_dim, self.n_dim])
            R[k] = multivariate_normal(mu, inv_lambda).reshape([self.n_dim, self.n_dim])
            numpy.mean(numpy.diag(inv_lambda))
            # logging.info('Mean variance R, %d, %f', k, mean_var)
        except:
            pass
# Helper functions to generate predictor classes.


def _logistic_regression() -> type:
    return from_class(sklearn.linear_model.LogisticRegression)


def _linear_regression() -> type:
    return from_class(sklearn.linear_model.LinearRegression, regression=True)


def _logistic_regression_committee() -> type:
    def make_committee(db, *args, **kwargs):
        return Committee(_logistic_regression(), db, *args, **kwargs)

    return make_committee


def _kde() -> type:
    return from_class(acton.kde_predictor.KDEClassifier)


PREDICTORS = {
    'LogisticRegression': _logistic_regression(),
    'LogisticRegressionCommittee': _logistic_regression_committee(),
    'LinearRegression': _linear_regression(),
    'KDE': _kde(),
    'GPC': GPClassifier,
    'TensorPredictor': TensorPredictor
}
