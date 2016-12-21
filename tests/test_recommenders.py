#!/usr/bin/env python3

"""
test_recommenders
----------------------------------

Tests for `recommenders` module.
"""

import unittest

import acton.recommenders


class TestRandomRecommender(unittest.TestCase):

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_recommend(self):
        """RandomRecommender recommends an instance."""
        ids = {str(a).encode('ascii') for a in range(1000)}
        rr = acton.recommenders.RandomRecommender(None)
        id_ = rr.recommend(ids, predictions=None)
        self.assertIn(id_[0], ids)
