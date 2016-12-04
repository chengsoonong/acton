"""Functions for reading/writing to protobufs."""

import struct
from typing import Union
from typing.io import BinaryIO

from google.protobuf.reflection import GeneratedProtocolMessageType
import numpy


def read_proto(
        path: str,
        Proto: GeneratedProtocolMessageType
) -> 'GeneratedProtocolMessageType()':
    """Reads a protobuf from a .proto file.

    Parameters
    ----------
    path
        Path to the .proto file.
    Proto:
        Protocol message class (from the generated protobuf module).

    Returns
    -------
    GeneratedProtocolMessageType
        The parsed protobuf.
    """
    proto = Proto()
    with open(path, 'rb') as proto_file:
        proto.ParseFromString(proto_file.read())

    return proto


def write_protos(path: str):
    """Serialises many protobufs to a file.

    Parameters
    ----------
    path
        Path to binary file. Will be overwritten.

    Notes
    -----
    Coroutine. Accepts protobufs, or None to terminate and close file.
    """
    with open(path, 'wb') as proto_file:
        proto = yield
        while proto:
            proto = proto.SerializeToString()
            # Protobufs are not self-delimiting, so we need to store the length
            # of each protobuf that we write. We will do this with an unsigned
            # long long (Q).
            length = struct.pack('<Q', len(proto))
            proto_file.write(length)
            proto_file.write(proto)
            proto = yield
    while True:
        proto = yield
        if proto:
            raise RuntimeError('Cannot write protobuf to closed file.')


def _read_protos(
        proto_file: BinaryIO,
        Proto: GeneratedProtocolMessageType
) -> 'GeneratedProtocolMessageType()':
    """Reads many protobufs from a file.

    Notes
    -----
    Internal use. For external API, use read_protos.

    Parameters
    ----------
    path
        Binary file.
    Proto:
        Protocol message class (from the generated protobuf module).

    Yields
    -------
    GeneratedProtocolMessageType
        A parsed protobuf.
    """
    # This is essentially the inverse of the write_protos function.
    length = proto_file.read(8)  # long long
    while length:
        length, = struct.unpack('<Q', length)
        proto = Proto()
        proto.ParseFromString(proto_file.read(length))
        yield proto
        length = proto_file.read(8)


def read_protos(
        file: Union[str, BinaryIO],
        Proto: GeneratedProtocolMessageType
) -> 'GeneratedProtocolMessageType()':
    """Reads many protobufs from a file.

    Parameters
    ----------
    file
        Path to binary file, or file itself.
    Proto:
        Protocol message class (from the generated protobuf module).

    Yields
    -------
    GeneratedProtocolMessageType
        A parsed protobuf.
    """
    try:
        yield from _read_protos(file, Proto)
    except AttributeError:
        # Not a file-like object, so open the file.
        with open(file, 'rb') as proto_file:
            yield from _read_protos(proto_file, Proto)


def get_ndarray(data: list, shape: tuple, dtype: str) -> numpy.ndarray:
    """Converts a list of values into an array.

    Parameters
    ----------
    data
        Raw array data.
    shape:
        Shape of the resulting array.
    dtype:
        Data type of the resulting array.

    Returns
    -------
    numpy.ndarray
        Array with the given data, shape, and dtype.
    """
    return numpy.array(data, dtype=dtype).reshape(tuple(shape))
