Acton - A scientific research assistant
=======================================

Acton is a modular Python library for active learning.
`Acton <https://en.wikipedia.org/wiki/Acton,_Australian_Capital_Territory>`__
is a suburb in Canberra, where Australian National University is
located.

|Build Status| |Documentation Status|

.. |Build Status| image:: https://travis-ci.org/chengsoonong/acton.svg?branch=master
   :target: https://travis-ci.org/chengsoonong/acton
.. |Documentation Status| image:: http://readthedocs.org/projects/acton/badge/?version=latest
   :target: http://acton.readthedocs.io/en/latest/?badge=latest

Dependencies
------------

Most dependencies will be installed by pip. You will need to manually install:

- Python 3.4+
- `Protobuf <https://github.com/google/protobuf/tree/master/python>`__

Setup
-----

Install Acton using ``pip3``:

.. code:: bash

    pip install git+https://github.com/chengsoonong/acton.git

This provides access to a command-line tool ``acton`` as well as the
``acton`` Python library.

Acton CLI
---------

The command-line interface to Acton is available through the ``acton``
command. This takes a dataset of features and labels and simulates an
active learning experiment on that dataset.

Input
+++++

Acton supports three formats of dataset: ASCII, pandas, and HDF5. ASCII
tables can be any file read by ``astropy.io.ascii.read``, including many common
plain-text table formats like CSV. pandas tables are supported if dumped to a
file from ``DataFrame.to_hdf``. HDF5 tables are either an HDF5 file with datasets
for each feature and a dataset for labels, or an HDF5 file with one
multidimensional dataset for features and one dataset for labels.

Output
++++++

Acton outputs a file containing predictions for each epoch of the simulation.
These are encoded as specified in `this notebook
<https://github.com/chengsoonong/acton/blob/master/docs/protobuf_spec.ipynb>`_.

Quickstart
----------

You will need a dataset. Acton currently supports ASCII tables (anything that can be read by :code:`astropy.io.ascii.read`), HDF5 tables, and Pandas tables saved as HDF5. `Here's a simple classification dataset <https://github.com/chengsoonong/acton/files/603416/classification.txt>`_ that you can use.

To run Acton to generate a passive learning curve with logistic regression:

.. code:: bash

    acton --data classification.txt --label col20 --feature col10 --feature col11 -o passive.pb --recommender RandomRecommender --predictor LogisticRegression

This command uses columns ``col10`` and ``col11`` as features, and ``col20`` as labels, a logistic regression predictor, and random recommendations. It outputs all predictions for test data points selected randomly from the input data to :code:`passive.pb`, which can then be used to construct a plot. To output an active learning curve using uncertainty sampling, change :code:`RandomRecommender` to :code:`UncertaintyRecommender`.

To show the learning curve, use `acton.plot`:

.. code:: bash

    python3 -m acton.plot passive.pb

Look at the directory ``examples`` for more examples.
