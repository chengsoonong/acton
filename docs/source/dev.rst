Developer Documentation
=======================

Contributing
------------

Adding a New Predictor
----------------------

Why Does Acton Use Predictor?
#############################

Acton makes use of ``Predictor`` classes, which are often just wrappers for scikit-learn classes. This raises the question: Why not just use scikit-learn classes?

This design decision was made because Acton must support predictors that do not fit the scikit-learn API, and so using scikit-learn predictors directly would mean that there is no unified API for predictors. An example of where Acton diverges from scikit-learn is that scikit-learn does not support multiple labellers.

Adding a New Recommender
------------------------
