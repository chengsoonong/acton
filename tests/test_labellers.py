#!/usr/bin/env python3

"""
test_labellers
----------------------------------

Tests for `labellers` module.
"""

import os.path
import sys
import tempfile
import unittest

import acton.labellers
import numpy


class TestASCIITableLabeller(unittest.TestCase):

    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()

        # Make an ASCII table. We'll just use part of the Norris 2006 catalogue
        # (Table 6, Norris et al. 2006).
        table = """\
name                      |ra         |dec        |is_agn
ATCDFS J032637.29-285738.2|03 26 37.30|-28 57 38.3|1     
ATCDFS J032642.55-285715.6|03 26 42.56|-28 57 15.7|1     
ATCDFS J032629.13-285648.7|03 26 29.13|-28 56 48.7|0     
ATCDFS J033056.94-285637.2|03 30 56.95|-28 56 37.3|0     
ATCDFS J033019.98-285635.5|03 30 19.98|-28 56 35.5|0     
ATCDFS J033126.71-285630.3|03 31 26.72|-28 56 30.3|0     
"""
        self.path = os.path.join(self.tempdir.name, 'table.dat')
        with open(self.path, 'w') as f:
            f.write(table)

    def tearDown(self):
        self.tempdir.cleanup()

    def test_query(self):
        """ASCIITableLabeller can query from a table."""
        labeller = acton.labellers.ASCIITableLabeller(
            self.path, 'name', 'is_agn')
        self.assertEqual(
            labeller.query(0),
            numpy.array([[1]]))
        self.assertEqual(
            labeller.query(4),
            numpy.array([[0]]))
