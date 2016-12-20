Acton - A scientific research assistant
=======================================

Active Learning: Predictors, Recommenders and Labellers

This is mostly a design document for a software system that uses active
learning, bandits and experimental design for suggesting items to label.

`Acton <https://en.wikipedia.org/wiki/Acton,_Australian_Capital_Territory>`__
is a suburb in Canberra, where Australian National University is
located.

.. image:: https://travis-ci.org/chengsoonong/acton.svg?branch=master
    :target: https://travis-ci.org/chengsoonong/acton

.. image:: http://readthedocs.org/projects/acton/badge/?version=latest
    :target: http://acton.readthedocs.io/en/latest/?badge=latest
    :alt: Documentation Status

Software tools
--------------

General
^^^^^^^

-  `cactus-flow <https://barro.github.io/2016/02/a-succesful-git-branching-model-considered-harmful/>`__
-  `cookiecutter <https://github.com/audreyr/cookiecutter-pypackage>`__
-  `click <http://click.pocoo.org/>`__
-  `attrs <https://attrs.readthedocs.io>`__

Serialization
^^^^^^^^^^^^^

-  `Astropy <http://www.astropy.org/>`__
-  `Protobuf <https://developers.google.com/protocol-buffers/docs/pythontutorial>`__

Architecture
------------

Three components, which interact with each other via serialised objects.

Predictors
^^^^^^^^^^

Wrapper to provide a common interface to external predictors. -
`sklearn <http://scikit-learn.org/>`__ - default predictors -
`keras <http://keras.io/>`__ - deep learning - `Vowpal
Wabbit <https://github.com/JohnLangford/vowpal_wabbit/wiki/Command-line-arguments>`__
- Command line interface - `GPy <https://sheffieldml.github.io/GPy/>`__
- Confidence intervals for regression

Labellers
^^^^^^^^^

Interfaces: - Read label from a file - Interactive human labellers for
RGZ like interface - Amazon mechanical turk

Potential applications: - cross identification of AGN - galaxy
classification - redshift estimation

Recommenders
^^^^^^^^^^^^

Get Alasdair to add code (to give appropriate credit) from
https://github.com/chengsoonong/mclass-sky

-  Batch recommendations, need to add some diversity
