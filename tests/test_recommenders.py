#!/usr/bin/env python3

"""
test_recommenders
----------------------------------

Tests for `recommenders` module.
"""

import numpy
import unittest
import unittest.mock

import acton.recommenders


class TestRandomRecommender(unittest.TestCase):

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_recommend(self):
        """RandomRecommender recommends an instance."""
        ids = set(range(1000))
        rr = acton.recommenders.RandomRecommender(None)
        id_ = rr.recommend(ids, predictions=None)
        self.assertIn(id_[0], ids)


class TestMarginRecommender(unittest.TestCase):

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_recommend(self):
        """MarginRecommender recommends an instance."""
        n = 10
        c = 3
        ids = list(range(n))
        predictions = numpy.random.random(size=(n, 1, c))
        db = unittest.mock.Mock()
        mr = acton.recommenders.MarginRecommender(db)
        id_ = mr.recommend(ids, predictions=predictions)
        self.assertIn(id_[0], ids)
