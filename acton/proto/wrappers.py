"""Classes that wrap protobufs."""

from typing import Union

import acton.database
import acton.proto.predictors_pb2 as predictors_pb
import acton.proto.io
import numpy


class PredictorInput(object):
    """Wrapper for the predictor input protobuf.

    Attributes
    ----------
    proto : predictors_pb.Labels
        Protobuf representing the input.
    """

    def __init__(self, proto: Union[str, predictors_pb.Labels]):
        """
        Parameters
        ----------
        proto
            Path to .proto file, or raw protobuf itself.
        """
        try:
            self.proto = acton.proto.io.read_proto(proto, predictors_pb.Labels)
        except TypeError:
            if isinstance(proto, predictors_pb.Labels):
                self.proto = proto
            else:
                raise TypeError('proto should be str or protobuf.')
        self._validate_proto()

    @property
    def labels(self) -> numpy.ndarray:
        """Gets labels array specified in input.

        Notes
        -----
        The returned array is cached by this object so future calls will not
        need to recompile the array.

        Returns
        -------
        numpy.ndarray
            T x N x F NumPy array of labels.
        """
        if hasattr(self, '_labels'):
            return self._labels

        self._labels = []
        for instance in self.proto.instance:
            data = instance.label
            shape = (self.proto.n_labellers, self.proto.n_label_dimensions)
            dtype = self.proto.dtype
            self._labels.append(acton.proto.io.get_ndarray(data, shape, dtype))
        self._labels = numpy.array(self._labels).transpose((1, 0, 2))
        return self._labels

    @property
    def features(self) -> numpy.ndarray:
        """Gets features array specified in input.

        Notes
        -----
        The returned array is cached by this object so future calls will not
        need to recompile the array.

        Returns
        -------
        numpy.ndarray
            N x D NumPy array of features.
        """
        if hasattr(self, '_features'):
            return self._features

        ids = [instance.id for instance in self.proto.instance]
        DB = acton.database.DATABASES[self.proto.db_class]
        with DB(self.proto.db_path, label_dtype=self.proto.dtype) as db:
            return db.read_features(ids)

    def _validate_proto(self):
        """Checks that the protobuf is valid and enforces constraints.

        Raises
        ------
        ValueError
        """
        if self.proto.db_class not in acton.database.DATABASES:
            raise ValueError('Invalid database class: {}'.format(
                self.proto.db_class))

        if self.proto.n_labellers < 1:
            raise ValueError('Number of labellers must be > 0.')

        if self.proto.n_label_dimensions < 1:
            raise ValueError('Label dimension must be > 0.')

        if not self.proto.db_path:
            raise ValueError('Must specify db_path.')

        if self.proto.n_label_dimensions > 1:
            raise NotImplementedError(
                'Multidimensional labels are not currently supported.')

    def _set_default(self):
        """Adds default parameters to the protobuf."""
        self.proto.dtype = self.proto.dtype or 'float32'
