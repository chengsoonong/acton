Acton - A scientific research assistant
=======================================

Acton is a modular Python library for active learning.
`Acton <https://en.wikipedia.org/wiki/Acton,_Australian_Capital_Territory>`__
is a suburb in Canberra, where Australian National University is
located.

|Build Status| |Documentation Status|

Setup
-----

Install
`Protobuf <https://github.com/google/protobuf/tree/master/python>`__.
Then install Acton using ``pip3``:

.. code:: bash

    pip install git+https://github.com/chengsoonong/acton.git

This provides access to a command-line tool ``acton`` as well as the
``acton`` Python library.

.. |Build Status| image:: https://travis-ci.org/chengsoonong/acton.svg?branch=master
   :target: https://travis-ci.org/chengsoonong/acton
.. |Documentation Status| image:: http://readthedocs.org/projects/acton/badge/?version=latest
   :target: http://acton.readthedocs.io/en/latest/?badge=latest

Quickstart
----------

You will need a dataset. Acton currently supports ASCII tables (anything that can be read by :code:`astropy.io.ascii.read`), HDF5 tables, and Pandas tables saved as HDF5. `Here's a simple classification dataset <https://github.com/chengsoonong/acton/files/603416/classification.txt>`_ that you can use.

To run Acton to generate a passive learning curve with logistic regression:

.. code:: bash

    acton classification.txt -o passive.pb --recommender RandomRecommender --predictor LogisticRegression

This outputs all predictions for test data points selected randomly from the input data to :code:`passive.pb`, which can then be used to construct a plot. To output an active learning curve using uncertainty sampling, change :code:`RandomRecommender` to :code:`UncertaintyRecommender`.
