.. Acton documentation master file, created by
   sphinx-quickstart on Sun Nov 13 15:48:45 2016.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to Acton's documentation!
=================================

Contents:

.. toctree::
   :maxdepth: 2

.. automodule:: acton
    :members:

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

Installation
====================

To install Acton, you will need protobuf-3.0.0. Then:

.. code:: bash
    git clone https://github.com/chengsoonong/acton.git
    cd acton
    protoc -I=acton/proto --python_out=acton/proto acton/proto/predictors.proto
    pip3 install -e .
