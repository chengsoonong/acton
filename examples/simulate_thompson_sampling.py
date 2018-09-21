# direct to parent folder
import sys
#sys.path.append("..")

from acton.database import LabelOnlyASCIIReader, LabelOnlyManagedHDF5Database
from acton.predictors import TensorPredictor
from acton.recommenders import ThompsonSamplingRecommender
from acton.labellers import LabelOnlyDatabaseLabeller
import acton.acton
import tempfile
import astropy.io.ascii as io_ascii
import os.path
import numpy as np
import logging

logging.basicConfig(level = logging.DEBUG)

_path = 'acton/tests/kg-data/nation/triples.txt'
output_path = 'acton/acton/acton.proto'
n_dim = 10

#data = io_ascii.read(_path)

#reader = LabelOnlyASCIIReader(_path, n_dim)
#reader.__enter__()


TS= 0.0
RANDOM = 1.0
N_EPOCHS = 100
repeated_labelling = False

with LabelOnlyASCIIReader(_path, n_dim) as reader:
    n_relations = reader.n_relations
    n_entities = reader.n_entities
    totoal_size = n_relations * n_entities * n_entities
    ids = np.arange(totoal_size)

    # TS
    TS_train_error_list, TS_test_error_list, TS_gain =  \
            acton.acton.simulate_active_learning(ids, reader, {}, output_path, 
                                                n_epochs= N_EPOCHS,
                                                recommender='ThompsonSamplingRecommender',
                                                predictor= 'TensorPredictor',
                                                labeller= 'LabelOnlyDatabaseLabeller',
                                                diversity= TS, 
                                                repeated_labelling = repeated_labelling)
    # Random
    RD_train_error_list, RD_test_error_list, RD_gain =  \
            acton.acton.simulate_active_learning(ids, reader, {}, output_path, 
                                                n_epochs= N_EPOCHS,
                                                recommender='ThompsonSamplingRecommender',
                                                predictor= 'TensorPredictor',
                                                labeller= 'LabelOnlyDatabaseLabeller',
                                                diversity= RANDOM,
                                                repeated_labelling = repeated_labelling)

acton.acton.plot(TS_train_error_list, TS_test_error_list, TS_gain, 
                 RD_train_error_list, RD_test_error_list, RD_gain)

