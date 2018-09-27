#!/usr/bin/env python3
"""Using Acton to test uncertainty sampling on the Australian libsvm dataset."""

import logging
import os.path
import tempfile

import acton.acton
import acton.plot
import h5py
import requests
import sklearn.datasets
import sklearn.preprocessing

with tempfile.TemporaryDirectory() as tempdir:
    # Download the dataset.
    # We'll store the dataset in this file:
    raw_filename = os.path.join(tempdir, 'australian.dat')
    dataset_response = requests.get(
        'https://www.csie.ntu.edu.tw/'
        '%7Ecjlin/libsvmtools/datasets/binary/australian')
    with open(raw_filename, 'w') as raw_file:
        raw_file.write(dataset_response.text)
    # Convert the dataset into a format we can use. It's currently libsvm.
    X, y = sklearn.datasets.load_svmlight_file(raw_filename)
    # We also have -1/1 as labels, but we need 0/1, so we'll convert.
    y = sklearn.preprocessing.LabelEncoder().fit_transform(y)
    # We'll just save it directly into an HDF5 file:
    input_filename = os.path.join(tempdir, 'australian.h5')
    with h5py.File(input_filename, 'w') as input_file:
        input_file.create_dataset('features', data=X.toarray())
        input_file.create_dataset('labels', data=y)

    # We'll save output to this file:
    output_base_filename = os.path.join(tempdir, 'australian_base.out')
    output_unct_filename = os.path.join(tempdir, 'australian_unct.out')

    # Run Acton.
    logging.root.setLevel(logging.DEBUG)
    acton.acton.main(
        data_path=input_filename,
        feature_cols=['features'],
        label_col='labels',
        output_path=output_base_filename,
        n_epochs=500,
        initial_count=100,
        recommender='RandomRecommender',
        predictor='LogisticRegression')
    acton.acton.main(
        data_path=input_filename,
        feature_cols=['features'],
        label_col='labels',
        output_path=output_unct_filename,
        n_epochs=500,
        initial_count=100,
        recommender='UncertaintyRecommender',
        predictor='LogisticRegression')

    # Plot the results.
    with open(output_base_filename, 'rb') as predictions_base, \
            open(output_unct_filename, 'rb') as predictions_unct:
        acton.plot.plot([predictions_base, predictions_unct])
