# Copyright 2019 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"). You
# may not use this file except in compliance with the License. A copy of
# the License is located at
#
#      http://aws.amazon.com/apache2.0/
#
# or in the "license" file accompanying this file. This file is
# distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF
# ANY KIND, either express or implied. See the License for the specific
# language governing permissions and limitations under the License.

import json
import os
from abc import ABC, abstractmethod
from sys import getsizeof

import mlio
from mlio.integ.numpy import as_numpy
import numpy as np
import psutil


def _convert_bytes_to_megabytes(b):
    """Converts bytes to megabytes"""
    return b / 1000 ** 2


def _convert_megabytes_to_bytes(mb):
    """Converts megabytes to bytes"""
    return mb * (1000 ** 2)


def _get_size_total(numpy_array):
    """Gets estimated memory usage of numpy array with dtype object in bytes."""
    assert numpy_array.dtype.kind == "O"
    assert 1 <= numpy_array.ndim <= 2
    total = numpy_array.nbytes  # Size of reference.
    rows = numpy_array if numpy_array.ndim == 2 else [numpy_array]
    for row in rows:
        total += sum([getsizeof(x) for x in row])
    return total


def _used_memory_mb():
    """Returns the current memory usage in megabytes"""
    return _convert_bytes_to_megabytes(psutil.virtual_memory().total - psutil.virtual_memory().available)


def _get_data(source):
    """Determines the input mode of the source and returns a InMemoryStore, SageMakerPipe, or File object
    based on the input mode.

    If source is a python buffer, a mlio.core.InMemoryStore will be returned.

    If SM_INPUT_DATA_CONFIG environment variable is not defined, source is assumed to be a file or directory and a
    mlio.core.File object will be returned.

    If SM_INPUT_DATA_CONFIG environment variable is defined, source can be the name of the channel in
    SM_INPUT_DATA_CONFIG. If the source is a path, it is assumed that the basename of the path is the name of the
    channel. The type of mlio.core object to be returned will be based on the "TrainingInputMode" of the channel.

    Here is an example of SM_INPUT_DATA_CONFIG with two channels ("code" and "train").
    SM_INPUT_DATA_CONFIG=
    {
        "code": {
            "ContentType": "application/x-code",
            "RecordWrapperType": "None",
            "S3DistributionType": "FullyReplicated",
            "TrainingInputMode": "File"
        },
        "train": {
            "ContentType": "text/csv",
            "RecordWrapperType": "None",
            "S3DistributionType": "ShardedByS3Key",
            "TrainingInputMode": "File"
        }
    }

    Parameters
    ----------
    source: str or bytes
        Name of the SageMaker Channel, File, or directory from which the data is being read or
        the Python buffer object from which the data is being read.

    Returns
    -------
    mlio.core.File:
        A mlio.core.File object is return based on the file or directory described by the `source`.

    mlio.core.SageMakerPipe:
        In SageMaker framework containers, the inputdataconfig.json is made available via environment
        variable 'SM_INPUT_DATA_CONFIG'. When the given source is a to 'Pipe' the value of the
        environment variable 'SM_INPUT_DATA_CONFIG' is used to read out the 'TrainingInputMode' and
        confirm that the source is a 'Pipe'. Then a `mlio.SageMakerPipe` object is created using the
        'source' and returned.

    mlio.core.InMemoryStore:
        Given the `source` is a Python buffer, a mlio.InMemoryStore object is created and returned
    """
    if isinstance(source, bytes):
        return [mlio.InMemoryStore(source)]

    if isinstance(source, mlio.core.File):
        source = source.id

    config = os.environ.get("SM_INPUT_DATA_CONFIG")

    if config is None:
        return mlio.list_files(source, pattern="*")

    channels = json.loads(config)

    source_channel_name = os.path.basename(source)
    try:
        channel_config = channels[source_channel_name]
    except KeyError:
        raise KeyError(
            "Configuration for channel name {} is not provided in SM_INPUT_DATA_CONFIG.".format(source_channel_name)
        )

    try:
        data_config_input_mode = channel_config["TrainingInputMode"]
    except KeyError:
        raise KeyError(
            "SM_INPUT_DATA_CONFIG is malformed. TrainingInputMode is "
            "not found for channel name {}".format(source_channel_name)
        )

    if data_config_input_mode == "Pipe":
        return [mlio.SageMakerPipe(source)]

    return mlio.list_files(source, pattern="*")  # 'File' mode


def _get_reader(source, batch_size):
    """Returns 'CsvReader' for the given source

       Parameters
       ----------
       source: str or bytes
           Name of the SageMaker Channel, File, or directory from which the data is being read or
           the Python buffer object from which the data is being read.

       batch_size : int
           The batch size in rows to read from the source.

       Returns
       -------
       mlio.CsvReader
           CsvReader configured with a SageMaker Pipe, File or InMemory buffer
       """
    return mlio.CsvReader(
        dataset=_get_data(source),
        batch_size=batch_size,
        default_data_type=mlio.DataType.STRING,
        header_row_index=None,
        allow_quoted_new_lines=True,
    )


class AbstractBatchConsumer(ABC):
    """Abstract utility class for consuming batches of columnar data up to a given size limit.

    Batches are recorded as numpy arrays as they come in, and concatenated into a final array when enough of them have
    been read so that that the resulting array is of size at most `max_size_in_bytes`, or the batches have been
    exhausted. This requires implementing an estimate of the size of the final array, which is done in the
    `_update_array_size_estimate` method.

    Parameters
    ----------
    max_size_in_bytes : int
        The maximum size that the resulting numpy array(s) should take up.

    target_column_index : int or None
        The index of the target column in the incoming batches. If present, column data is split into a separate
        array than the remaining "feature" data.
    """

    def __init__(self, max_size_in_bytes, target_column_index=None):
        self.max_size_in_bytes = max_size_in_bytes
        self.target_column_index = target_column_index

        # Lists which hold the batch data to concatenate in the end.
        self._features_batches = []
        self._target_batches = []

        # Number of columns to be inferred from first batch.
        self._n_columns = 0

        # State tracking convenience variables.
        self._consumed = False
        self._initialized = False
        self._split_target = target_column_index is not None

    def _initialize_state(self, first_batch):
        self._n_columns = len(first_batch)
        if self._split_target:
            assert (
                self.target_column_index < self._n_columns
            ), f"Invalid target_column_index {self.target_column_index} in data with {self._n_columns} columns."
            # Enable loading single-column datasets with self.target_column_index set.
            if self._n_columns == 1:
                self.target_column_index = None
                self._split_target = False

        self._initialized = True

    def _add_batch(self, batch):
        """Adds batch or truncated batch to the concatenation list.

        A batch is truncated if adding it would make the final array exceed the maximum target size.

        Parameters
        ----------
        batch : mlio Example
            An MLIO batch returned by CsvReader, with data encoded as strings.as an Example class. Can be used to
             easily iterate over columns of data in batch.

        Returns
        -------
        bool
            True if adding batches should continue, False otherwise.
        """
        # Perform initialization on first batch.
        if not self._initialized:
            self._initialize_state(batch)

        # Construct numpy representation for data in batch.
        features_array_data = self._construct_features_array_data(batch)
        target_array_data = self._construct_target_array_data(batch)

        # Update size estimation variables.
        batch_nbytes_estimate, total_nbytes_estimate = self._update_array_size_estimate(
            features_array_data, target_array_data
        )

        # If the resulting array will be too large, truncate the last batch so that it fits.
        should_continue = True
        if total_nbytes_estimate > self.max_size_in_bytes:
            batch_bytes_to_keep = batch_nbytes_estimate - (total_nbytes_estimate - self.max_size_in_bytes)
            fraction_of_batch_rows_to_keep = batch_bytes_to_keep / batch_nbytes_estimate
            n_rows_to_keep = int(fraction_of_batch_rows_to_keep * self._n_rows(features_array_data))

            if n_rows_to_keep > 0:
                features_array_data = self._resize_features_array_data(features_array_data, n_rows_to_keep)
                if self._split_target:
                    target_array_data = self._resize_target_array_data(target_array_data, n_rows_to_keep)

            should_continue = False

        self._extend_features_batches(features_array_data)
        if self._split_target:
            self._extend_target_batches(target_array_data)

        return should_continue

    def consume_reader(self, reader):
        """Reads batches from reader and returns array of size less than or equal to the indicated limit.

        Parameters
        ----------
        reader : mlio.CsvReader
            A new reader instance from which to load the data.

        Returns
        -------
        numpy array or tuple of numpy arrays
            A single numpy array is returned when `target_column_index` is None.

            A tuple of numpy arrays is returned when `target_column_index` is not None. The first element of the tuple
            is a 2D numpy array containing all columns except the one corresponding to `target_column_index`.
            The second element of the tuple is 1D numpy array corresponding to `target_column_index`.
        """
        if self._consumed:
            raise RuntimeError("This instance has already been used to consume a batch reader.")

        for batch in reader:
            should_continue = self._add_batch(batch)
            if not should_continue:
                break

        self._consumed = True
        return self._concatenate_data()

    @abstractmethod
    def _concatenate_data(self):
        """Concatenates recorded batches into final numpy array(s).

        Returns
        -------
        numpy array or tuple of numpy arrays
            A single numpy array is returned when `target_column_index` is None.

            A tuple of numpy arrays is returned when `target_column_index` is not None. The first element of the tuple
            is a 2D numpy array containing all columns except the one corresponding to `target_column_index`.
            The second element of the tuple is 1D numpy array corresponding to `target_column_index`.
        """

    @abstractmethod
    def _construct_features_array_data(self, batch):
        """Constructs a data structure containing feature column numpy arrays from the batch.

        Parameters
        ----------
        batch : mlio Example
            An MLIO batch returned by CsvReader, with data encoded as strings.as an Example class. Can be used to
             easily iterate over columns of data in batch.

        Returns
        -------
        Feature data encoded as np.array(s).
        """

    @abstractmethod
    def _construct_target_array_data(self, batch):
        """Constructs a numpy array with the target column data in the batch.

        Parameters
        ----------
        batch : mlio batch as an Example class -- can be used to easily iterate over columns of data in batch.

        Returns
        -------
        np.array or None
            None is returned if `target_column_index` is None.
        """

    @abstractmethod
    def _extend_features_batches(self, features_array_data):
        """Saves `features_array_data` into `self._features_batches`.

        Parameters
        ----------
        features_array_data : np.array or list of np.array
            Feature columns from data batch processed into numpy array(s).

        Returns
        -------
        None
        """

    @abstractmethod
    def _extend_target_batches(self, target_array_data):
        """Saves `target_array_data` into `self._target_batches`.

        Parameters
        ----------
        target_array_data: np.array or None
            Target column from data batch processed into numpy array. Can be None if the `target_column_index` parameter
            has not been specified.

        Returns
        -------
        None
        """

    @abstractmethod
    def _n_rows(self, features_array_data):
        """Returns the number of rows in `features_array_data`.

        Parameters
        ----------
        features_array_data : np.array or list of np.array
            Feature columns from data batch processed into numpy array(s).

        Returns
        -------
        int
            Number of rows in incoming batch.
        """

    @abstractmethod
    def _resize_features_array_data(self, features_array_data, n_rows_to_keep):
        """Truncates feature numpy array data to `n_rows_to_keep`.

        Parameters
        ----------
        features_array_data : np.array or list of np.array
            Feature columns from data batch processed into numpy array(s).

        n_rows_to_keep : int

        Returns
        -------
        Truncated feature numpy array data.
        """

    def _resize_target_array_data(self, target_array_data, n_rows_to_keep):
        """Truncates target numpy array to `n_rows_to_keep`

        Parameters
        ----------
        target_array_data: np.array or None
            Target column from data batch processed into numpy array. Can be None if the `target_column_index` parameter
            has not been specified.

        n_rows_to_keep : int

        Returns
        -------
        np.array
            Truncated array slice.
        """
        return target_array_data[:n_rows_to_keep]

    @abstractmethod
    def _update_array_size_estimate(self, features_array_data, target_array_data):
        """Updates internal state required to estimate the size of the final array.

        This estimate will vary depending on the storage mechanism of the batches as they are being read, and the
        format of the final array.

        Parameters
        ----------
        features_array_data : np.array or list of np.array
            Feature columns from data batch processed into numpy array(s).

        target_array_data: np.array or None
            Target column from data batch processed into numpy array. Can be None if the `target_column_index` parameter
            has not been specified.

        Returns
        -------
        tuple of ints
            Tuple consisting of estimated size of batch in final array, and estimated total size of final array. Both
            values are returned in bytes.
        """


class ObjectBatchConsumer(AbstractBatchConsumer):
    """Utility class which reads incoming batches as-is and concatenates them without casting into a specific dtype.

    Since batches come in with dtype object (that's what `default_data_type=mlio.DataType.STRING` does in `_get_reader`
    above), the result will also be an array with dtype object.
    """

    def __init__(self, max_size_in_bytes, target_column_index=None):
        super().__init__(max_size_in_bytes, target_column_index)

        # Average amount of memory in one row (references included), estimated from the first batch.
        self._row_nbytes = 0
        self._estimated_size_in_bytes = 0

    def _initialize_state(self, first_batch):
        super()._initialize_state(first_batch)

        # Estimate the size of items in each column using the first batch.
        for i in range(self._n_columns):
            column = as_numpy(first_batch[i]).flatten()
            self._row_nbytes += _get_size_total(column) / column.shape[0]

    def _concatenate_data(self):
        """Concatenates feature and target data arrays into the final array(s)."""
        if self._features_batches:
            feature_data = np.concatenate(self._features_batches)
        else:
            feature_data = np.array([])

        if self._split_target:
            if self._target_batches:
                target_data = np.concatenate(self._target_batches)
            else:
                target_data = np.array([])
            return feature_data, target_data
        return feature_data

    def _construct_features_array_data(self, batch):
        """Stacks numpy columns created from an incoming data batch into a numpy array."""
        return np.column_stack(
            [
                as_numpy(batch[column_index]).flatten()
                for column_index in range(self._n_columns)
                if column_index != self.target_column_index
            ]
        )

    def _construct_target_array_data(self, batch):
        if self._split_target:
            return as_numpy(batch[self.target_column_index]).flatten()
        return None

    def _extend_features_batches(self, features_array_data):
        """Appends the numpy array created from an incoming batch to the features batch list."""
        self._features_batches.append(features_array_data)

    def _extend_target_batches(self, target_array_data):
        """Appends the numpy array created from an incoming batch to the target batch list."""
        self._target_batches.append(target_array_data)

    def _n_rows(self, features_array_data):
        """Returns the number of rows in feature data extracted from batch. """
        return features_array_data.shape[0]

    def _resize_features_array_data(self, features_array_data, n_rows_to_keep):
        """Truncates the incoming feature data batch to a length of `n_rows_to_keep`."""
        return features_array_data[:n_rows_to_keep]

    def _update_array_size_estimate(self, features_array_data, target_array_data):
        """Estimates the size of the final dataset using row sizes obtained from first batch."""
        batch_nbytes = self._row_nbytes * self._n_rows(features_array_data)
        self._estimated_size_in_bytes += batch_nbytes
        return batch_nbytes, self._estimated_size_in_bytes


class StringBatchConsumer(AbstractBatchConsumer):
    """Utility class which reads incoming batches and returns the final array(s) with dtype string.

    This class consumes batches produced by MLIO's CsvReader. As each batch is consumed, we estimate the size of the
    array with dtype string that would be produced by concatenating the batches so far. That is then compared against
    the limit in bytes to determine whether to stop consuming the batches.

    Note that memory usage might be smaller than `max_size_in_bytes`, because the final array's size estimate is based
    on the max itemsize encountered in any batch, and a portion of the last batch may be discarded.

    `self._features_batches` is a list of lists. Each sublist corresponds to a feature column in the input
    dataset, and contains numpy arrays of the data that came in for that column in each batch.
    """

    def __init__(self, max_size_in_bytes, target_column_index=None):
        super().__init__(max_size_in_bytes, target_column_index)

        # Total number of items loaded so far.
        self._features_size = 0
        self._target_size = 0

        # Maximum itemsizes encountered so far.
        self._max_features_itemsize = 0
        self._target_itemsize = 0

    def _initialize_state(self, first_batch):
        super()._initialize_state(first_batch)

        # The number of feature (non-target) columns.
        self._n_features = self._n_columns - (1 if self._split_target else 0)

        # self._features_batches[i] contains numpy arrays containing the data from feature column i in each batch.
        for _ in range(self._n_features):
            self._features_batches.append([])

        # Maintain a separate itemsize for each column.
        self._features_itemsizes = [0 for _ in range(self._n_features)]

    def _concatenate_data(self):
        """Concatenates individual columns, and stacks them into a larger array."""
        # Replace batched columns in `self._features_batches` with a concatenated version one at a time.
        if self._features_batches and self._features_batches[0]:
            for i in range(self._n_features):
                self._features_batches[i] = np.concatenate(self._features_batches[i])
            features_data = np.column_stack(self._features_batches)
        else:
            features_data = np.array([]).astype(str)

        if self._split_target:
            if self._target_batches:
                target_data = np.concatenate(self._target_batches)
            else:
                target_data = np.array([])
            return features_data, target_data
        return features_data

    def _construct_features_array_data(self, batch):
        """Creates a list of `self._n_features` arrays containing data from each column in the batch.

        Note that the arrays are interpreted as strings here, in order to easily extract itemsize and estimate size.
        """
        return [
            as_numpy(batch[i]).flatten().astype(str) for i in range(self._n_columns) if i != self.target_column_index
        ]

    def _construct_target_array_data(self, batch):
        if self._split_target:
            return as_numpy(batch[self.target_column_index]).flatten().astype(str)
        return None

    def _extend_features_batches(self, features_array_data):
        """Appends the numpy arrays created from an incoming batch to the features batch list."""
        for i, column_batch in enumerate(features_array_data):
            self._features_batches[i].append(column_batch)

    def _extend_target_batches(self, target_array_data):
        """Appends the target numpy array created from an incoming batch to the target batch list."""
        self._target_batches.append(target_array_data)

    def _n_rows(self, features_array_data):
        """Returns the number of rows in feature data extracted from batch. """
        return features_array_data[0].shape[0] if features_array_data else 0

    def _resize_features_array_data(self, features_array_data, n_rows_to_keep):
        """Truncates each feature's incoming data to a length of `n_rows_to_keep`."""
        return [column[:n_rows_to_keep] for column in features_array_data]

    def _update_array_size_estimate(self, features_array_data, target_array_data):
        """Estimates the size of the final array when the incoming array data is added to it."""
        feature_batch_size = 0
        for i in range(self._n_features):
            feature_column_array = features_array_data[i]
            self._features_itemsizes[i] = max(self._features_itemsizes[i], feature_column_array.itemsize)
            self._max_features_itemsize = max(self._features_itemsizes[i], self._max_features_itemsize)
            feature_batch_size += feature_column_array.size
        self._features_size += feature_batch_size

        batch_size_in_bytes = feature_batch_size * self._max_features_itemsize
        total_size_in_bytes = self._features_size * self._max_features_itemsize

        if self._split_target:
            self._target_itemsize = max(target_array_data.itemsize, self._target_itemsize)
            self._target_size += target_array_data.size
            batch_size_in_bytes += target_array_data.size * self._target_itemsize
            total_size_in_bytes += self._target_itemsize * self._target_size
        return batch_size_in_bytes, total_size_in_bytes


def _read_to_fit_memory(reader, max_memory_bytes, target_column_index=None, output_dtype="O"):
    """Reads batches from reader until a numpy array of size up to `max_memory_bytes` is returned.

    The array will dtype.kind 'U' if output_dtype is 'U', and dtype.kind 'O' otherwise.

    Parameters
    ----------

    reader : mlio.CsvReader
        MLIO reader yielding data batches as Examples -- collections of tensors that can be cast to numpy arrays.

    max_memory_mb : int
        Maximum total memory usage in bytes of the returned array(s).

    target_column_index : int or None
        Index of target column in the input dataset. If not None, data in the corresponding column of the CSV being
        read will be separated into a 1D numpy array.

    output_dtype : string
        If this value is 'U', then the returned numpy array(s) will have dtype.kind = 'U'. Otherwise,
        the return array(s) will have dtype.kind = 'O'.

    Returns
    -------
    numpy array or tuple of numpy arrays
        A single numpy array is returned when `target_column_index` is None.

        A tuple of numpy arrays is returned when `target_column_index` is not None. The first element of the tuple
        is a 2D numpy array containing all columns except the one corresponding to `target_column_index`.
        The second element of the tuple is 1D numpy array corresponding to `target_column_index`.
    """
    if output_dtype == "U":
        reader_consumer = StringBatchConsumer(max_memory_bytes, target_column_index)
    else:
        reader_consumer = ObjectBatchConsumer(max_memory_bytes, target_column_index)
    return reader_consumer.consume_reader(reader)


def read_csv_data(
    source: str or bytes,
    batch_size: int = 1000,
    fit_memory_percent: float = 20.0,
    target_column_index: int = None,
    output_dtype: str = "O",
):
    """Reads comma separated data and returns a tuple of numpy arrays.

    This function reads the csv data from either a `SageMakerPipe`, `File`, or `InMemoryStore` buffer.
    If `fit_memory_percent` is set to a positive threshold, it identifies the number of samples that can be loaded to
    fit in the requested percentage of the memory.

    Parameters
    -------
    source : str or bytes
        The source must correspond to one of the following:

        'File':
            This should be used if data is being read through a file or directory. If used, the
            'source' should be the file or directory's path.

        'Pipe':
            A 'Pipe' streams data directly from Amazon S3 to a container. If a 'Pipe' is used,
            the 'source' should be the name of the desired SageMaker channel.

            For more information on 'Pipe' mode see:
            https://aws.amazon.com/blogs/machine-learning/using-pipe-input-mode-for-amazon-sagemaker-algorithms/

       'InMemory':
           This should be used when the data is being read in bytes through an in-memory Python buffer.
           If used, 'source' should be the Python buffer object.

    batch_size : int
        The batch size in rows to read from the source.

    fit_memory_percent : float
        Sample down the examples to use the maximum percentage of the available memory.

    target_column_index : int or None
        Index of target column in the input dataset. If not None, data in the corresponding column of the CSV being
        read will be separated into a 1D numpy array.

    output_dtype : string
        If this value is 'U', then the returned numpy array(s) will have dtype.kind = 'U'. Otherwise,
        the return array(s) will have dtype.kind = 'O'.

    Returns
    -------
    numpy array or tuple of numpy arrays
        A single numpy array is returned when `target_column_index` is None.

        A tuple of numpy arrays is returned when `target_column_index` is not None. The first element of the tuple
        is a 2D numpy array containing all columns except the one corresponding to `target_column_index`.
        The second element of the tuple is 1D numpy array corresponding to `target_column_index`.
    """
    max_memory_bytes = psutil.virtual_memory().total * (fit_memory_percent / 100)

    return _read_to_fit_memory(
        _get_reader(source, batch_size),
        max_memory_bytes,
        target_column_index=target_column_index,
        output_dtype=output_dtype,
    )
