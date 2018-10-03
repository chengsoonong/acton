#!/usr/bin/env python3

"""
test_proto_io
----------------------------------

Tests for `proto.io` module.
"""

import os.path
import tempfile
from typing import List
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

    @staticmethod
    def make_protobufs(n: int=10) -> List[bytes]:
        """Makes random-length "protobufs".

        Parameters
        ----------
        n
            Number of protobufs.

        Yields
        ------
        bytes
            A random protobuf.
        """
        for _ in range(n):
            protobuf = bytes(
                numpy.random.randint(256)
                for _ in range(numpy.random.randint(100)))
            yield protobuf

    def test_write_read(self):
        """Protobufs written by write_protos can be read by read_protos."""
        # We will assume that the write/read functions are using the
        # serialisation methods built-in to protobufs. We thus deal only with
        # serialised protobufs.
        serialised_protos = list(self.make_protobufs())

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

    def test_write_read_file(self):
        """read_protos accepts opened binary files."""
        # This function is identical to test_write_read but with files instead
        # of paths.

        serialised_protos = list(self.make_protobufs())

        # Write the protobufs.
        writer = acton.proto.io.write_protos(self.path)
        next(writer)
        for protobuf in serialised_protos:
            self.proto.SerializeToString.return_value = protobuf
            writer.send(self.proto)
        writer.send(None)

        # Read the protobufs using a file object.
        with open(self.path, 'rb') as proto_file:
            read_protobufs = [i.proto for i in acton.proto.io.read_protos(
                proto_file, self.Proto)]
            self.assertEqual(serialised_protos, read_protobufs)
