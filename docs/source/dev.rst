Developer Documentation
=======================

Contributing
------------

We accept pull requests on GitHub. Contributions must be PEP8 compliant and pass
formatting and function tests in the test script ``/test``.

Adding a New Predictor
----------------------

A predictor is a class that implements ``acton.predictors.Predictor``. Adding a
new predictor amounts to implementing a subclass of ``Predictor`` and
registering it in ``acton.predictors.PREDICTORS``.

Predictors must implement:

- ``__init__(db: acton.database.Database, *args, **kwargs)``, which stores a reference to the database (and does any other initialisation).
- ``fit(ids: Iterable[int])``, which takes an iterable of IDs and fits a model
  to the associated features and labels,
- ``predict(ids: Sequence[int]) -> numpy.ndarray``, which takes a sequence of
  IDs and predicts the associated labels.
- ``reference_predict(ids: Sequence[int]) -> numpy.ndarray``, which behaves the same as ``predict`` but uses the best possible model.

Predictors should store data-based values such as the model in attributes ending in an underscore, e.g. ``self.model_``.

Why Does Acton Use Predictor?
#############################

Acton makes use of ``Predictor`` classes, which are often just wrappers for
scikit-learn classes. This raises the question: Why not just use scikit-learn
classes?

This design decision was made because Acton must support predictors that do not
fit the scikit-learn API, and so using scikit-learn predictors directly would
mean that there is no unified API for predictors. An example of where Acton
diverges from scikit-learn is that scikit-learn does not support multiple
labellers.

Adding a New Recommender
------------------------

A recommender is a class that implements ``acton.recommenders.Recommender``. Adding a new recommender amounts to implementing a subclass of ``Recommender`` and registering it in ``acton.recommenders.RECOMMENDERS``.

Recommenders must implement:

- ``__init__(db: acton.database.Database, *args, **kwargs)``, which stores a reference to the database (and does any other initialisation).
- ``recommend(ids: Iterable[int], predictions: numpy.ndarray, n: int=1, diversity: float=0.5)` -> Sequence[int]``, which recommends ``n`` IDs from the given IDs based on the associated predictions.
