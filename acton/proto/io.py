"""Functions for reading/writing to protobufs."""

from google.protobuf.reflection import GeneratedProtocolMessageType
import numpy


def read_proto(path: str, Proto: GeneratedProtocolMessageType) -> GeneratedProtocolMessageType():
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
