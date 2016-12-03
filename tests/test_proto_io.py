#!/usr/bin/env python3

"""
test_proto_io
----------------------------------

Tests for `proto.io` module.
"""

import os.path
import sys
import tempfile
import unittest
import unittest.mock

import acton.proto.io
import numpy


class TestIOMany(unittest.TestCase):
    """Tests read_protos and write_protos."""

    def setUp(self):
        # Make some (mock) protobufs.
        self.proto = unittest.mock.Mock(['SerializeToString'])
        self.tempdir = tempfile.TemporaryDirectory()
        self.path = os.path.join(self.tempdir.name, 'testiomany.proto')
        # And a mock class for deserialisation.
        class Proto:
            def ParseFromString(self, string):
                self.proto = string
        self.Proto = Proto
        # The idea here is to store the serialised protobuf, so we can check
        # that it has been read in correctly.

    def tearDown(self):
        self.tempdir.cleanup()

    def test_write_read(self):
        """Protobufs written by write_protos can be read by read_protos."""
        # We will assume that the write/read functions are using the
        # serialisation methods built-in to protobufs. We thus deal only with
        # serialised protobufs.
        serialised_protos = []
        # Make some random-length "protobufs".
        for _ in range(10):
            protobuf = bytes(
                numpy.random.randint(256)
                for _ in range(numpy.random.randint(100)))
            serialised_protos.append(protobuf)

        # Write the protobufs.
        writer = acton.proto.io.write_protos(self.path)
        next(writer)
        for protobuf in serialised_protos:
            self.proto.SerializeToString.return_value = protobuf
            writer.send(self.proto)
        writer.send(None)

        # Read the protobufs.
        read_protobufs = [i.proto
            for i in acton.proto.io.read_protos(self.path, self.Proto)]
        self.assertEqual(serialised_protos, read_protobufs)
