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

import psutil

import csv
from contextlib import contextmanager
import json
import numpy as np
import os
import pytest

from mlio import list_files
from mlio.core import InMemoryStore, SageMakerPipe
from mlio.core import InvalidInstanceError
from mlio.core import File as mlio_file
from sagemaker_sklearn_extension.externals.read_data import _convert_megabytes_to_bytes
from sagemaker_sklearn_extension.externals.read_data import _get_data
from sagemaker_sklearn_extension.externals.read_data import _get_reader
from sagemaker_sklearn_extension.externals.read_data import _get_size_total
from sagemaker_sklearn_extension.externals.read_data import _read_to_fit_memory
from sagemaker_sklearn_extension.externals.read_data import read_csv_data


DATA_FILES = [
    "test/data/csv/mock_datasplitter_output/manual.csv",
    "test/data/csv/mock_datasplitter_output/newline.csv",
    "test/data/csv/mock_datasplitter_output/excel.csv",
    "test/data/csv/mock_datasplitter_output/oneline.csv",
    "test/data/csv/missing_values.csv",
    "test/data/csv/dictionaries.csv",
    "test/data/csv/dirty.csv",
]
DATA_FILES_SHAPE = [(8, 4), (10, 4), (3, 4), (1, 4), (7, 5), (147, 18), (19, 16)]
LARGE_DATA_4MB = "test/data/csv/kc_house_data.csv"
BUFFER_DATA = (
    "1,2,3,4\n"
    + "5,6,7,8\n"
    + "9,10,11,12\n"
    + "13,14,15,16\n"
    + "17,18,19,20\n"
    + "21,22,23,24\n"
    + "25,26,27,28\n"
    + "29,30,31,32"
)


@contextmanager
def managed_env_var(cfg):
    os.environ.update({"SM_INPUT_DATA_CONFIG": json.dumps(cfg)})
    try:
        yield os.environ
    finally:
        os.environ.pop("SM_INPUT_DATA_CONFIG")


csv1 = [
    ["1.0", 2.0, "3", 4, ""],
    ["a,b", "c\nd", "f", '"""', np.nan],
]
csv2 = [
    [10, "2\r\n4", "hello", 4.0, "!"],
    [" space", "", "space ", "\n", "hello\n"],
    ['{a: 5, b: "hello"}', "[a, b, 2]", "[]", "nan", " "],
]


@pytest.fixture(scope="session")
def csv_data_dir(tmpdir_factory):
    """Fixture which fills a temporary directory with (multiple) csv file(s)."""
    csv_data_directory = tmpdir_factory.mktemp("csv_file_paths")
    csv_file1 = csv_data_directory.join("file_1.csv")
    csv_file2 = csv_data_directory.join("file_2.csv")

    with open(csv_file1.strpath, "w") as csv_file_handle:
        csv_writer = csv.writer(csv_file_handle, dialect="excel")
        csv_writer.writerows(csv1)
    with open(csv_file2.strpath, "w") as csv_file_handle:
        csv_writer = csv.writer(csv_file_handle, dialect="excel")
        csv_writer.writerows(csv2)

    return str(csv_data_directory)


def test_excel_dialect(csv_data_dir):
    """Test that read_csv_data function properly reads files in the excel dialect."""
    generated_contents = read_csv_data(source=csv_data_dir + "/file_1.csv")

    assert generated_contents.shape == (len(csv1), len(csv1[0]))
    assert np.all(generated_contents == np.array([[str(v) for v in row] for row in csv1], dtype=np.str))


def test_directory_content(csv_data_dir):
    """Test that read_csv_data function reads content correctly from a directory"""
    generated_contents = read_csv_data(source=csv_data_dir)
    correct_array = csv1 + csv2
    assert generated_contents.shape == (len(correct_array), len(correct_array[0]))
    assert np.all(generated_contents == np.array([[str(v) for v in row] for row in correct_array], dtype=np.str))


def test_get_reader_pipe_mode():
    """Test for getting a 'CsvReader' object with 'Pipe' mode"""
    with managed_env_var({"abc": {"TrainingInputMode": "Pipe"}}):
        reader = _get_data(source="abc")
        assert isinstance(reader[0], SageMakerPipe)


def test_get_reader_file_mode():
    """Test for getting a 'CsvReader' object with 'File' mode"""
    source = "test/data/csv/mock_datasplitter_output"
    with managed_env_var({os.path.basename(source): {"TrainingInputMode": "File"}}):
        reader = _get_data(source=source)
        assert isinstance(reader[0], mlio_file)


def test_get_reader_mlio_file_object():
    """Test for getting a 'CsvReader' with a mlio.core.File object source"""
    source = "test/data/csv/mock_datasplitter_output"
    files = list_files(source, pattern="*")
    reader = _get_data(source=files[0])
    assert isinstance(reader[0], mlio_file)


def test_get_reader_inmemory_mode():
    """Test for getting a 'CsvReader' object with 'InMemory' mode"""
    buffer = BUFFER_DATA.encode()
    reader = _get_data(source=buffer)
    assert isinstance(reader[0], InMemoryStore)


def test_read_csv_data_inmemory_mode():
    """Test to make sure 'InMemory' mode reads in content correctly"""
    generated_contents = read_csv_data(source=BUFFER_DATA.encode())
    correct_array = []
    for i in range(8):
        correct_array.append([i * 4 + j for j in range(1, 5)])
    assert generated_contents.shape == (len(correct_array), len(correct_array[0]))
    assert np.all(generated_contents == np.array([[str(v) for v in row] for row in correct_array], dtype=np.str))


def test_read_empty_buffer():
    """Test for getting an empty array if the buffer is empty"""
    generated_contents = read_csv_data(source="".encode())
    assert generated_contents.size == 0


def test_get_reader_no_env_var():
    """Test for getting a 'CsvReader' object with no environmental variable"""
    reader = _get_data(source="test/data/csv/mock_datasplitter_output")
    assert isinstance(reader[0], mlio_file)


@pytest.mark.parametrize("cfg, expected_error", [({}, KeyError), ({"abc": {}}, KeyError),])
def test_get_reader_error_malformed_channel_cfg(cfg, expected_error):
    """Test for reading from an invalid channel"""
    with pytest.raises(expected_error):
        with managed_env_var(cfg):
            _get_reader(source="abc", batch_size=1000)


def test_get_reader_incorrect_path():
    """Test for reading from a path that doesn't exist"""
    with pytest.raises(RuntimeError):
        _get_reader(source="incorrect", batch_size=100)


def test_read_csv_data_invalid_csv():
    with pytest.raises(InvalidInstanceError):
        read_csv_data(source="test/data/csv/invalid.csv")


@pytest.mark.parametrize("data_file, shape", [(file, shape) for file, shape in zip(DATA_FILES, DATA_FILES_SHAPE)])
def test_read_csv_data(data_file, shape):
    """Test for reading individual csv data files"""
    array = read_csv_data(source=data_file, batch_size=1, fit_memory_percent=100.0, output_dtype="U")
    assert array.shape == shape
    assert array.dtype.kind in {"U", "S"}


def test_read_csv_data_directory():
    """Test for reading from a directory of data"""
    array = read_csv_data(source="test/data/csv/mock_datasplitter_output", fit_memory_percent=100.0)
    assert array.shape == (22, 4)


def test_read_csv_data_sample_append():
    """Test for reading data in chunks."""
    array = read_csv_data(source=LARGE_DATA_4MB, fit_memory_percent=100.0)
    assert array.shape == (38223, 21)


def test_read_csv_data_samples():
    """Test for sample case where the entire dataset doesn't fit into the available memory"""
    total_memory_in_bytes = psutil.virtual_memory().total
    two_mb_in_bytes = _convert_megabytes_to_bytes(2)
    fraction_of_memory_to_use = two_mb_in_bytes / total_memory_in_bytes
    sample_data = read_csv_data(
        source=LARGE_DATA_4MB, fit_memory_percent=fraction_of_memory_to_use * 100, output_dtype="U"
    )
    assert sample_data.dtype.kind == "U"
    assert _convert_megabytes_to_bytes(1.9) < sample_data.nbytes <= two_mb_in_bytes


def test_read_csv_data_split():
    X, y = read_csv_data(LARGE_DATA_4MB, target_column_index=0, output_dtype="U")
    yX = read_csv_data(LARGE_DATA_4MB, output_dtype="U")
    assert X.shape == (38223, 20)
    assert y.shape == (38223,)
    assert np.array_equal(np.hstack((y.reshape(-1, 1), X)).astype(str), yX)
    assert X.dtype.kind == "U"
    assert y.dtype.kind == "U"


def test_read_csv_data_split_limited():
    total_memory_in_bytes = psutil.virtual_memory().total
    two_mb_in_bytes = _convert_megabytes_to_bytes(2)
    fraction_of_memory_to_use = two_mb_in_bytes / total_memory_in_bytes
    X, y = read_csv_data(
        LARGE_DATA_4MB, target_column_index=0, fit_memory_percent=fraction_of_memory_to_use * 100, output_dtype="U"
    )
    assert _convert_megabytes_to_bytes(1.9) < (X.nbytes + y.nbytes) <= two_mb_in_bytes
    assert X.dtype.kind == "U"
    assert y.dtype.kind == "U"


def test_read_csv_data_samples_object():
    """Test for sample case where the entire dataset doesn't fit into the available memory"""
    total_memory_in_bytes = psutil.virtual_memory().total
    two_mb_in_bytes = _convert_megabytes_to_bytes(2)
    fraction_of_memory_to_use = two_mb_in_bytes / total_memory_in_bytes
    sample_data = read_csv_data(
        source=LARGE_DATA_4MB, fit_memory_percent=fraction_of_memory_to_use * 100, output_dtype="object"
    )
    array_memory = _get_size_total(sample_data)
    assert _convert_megabytes_to_bytes(1.9) < array_memory <= two_mb_in_bytes
    assert sample_data.dtype.kind == "O"


def test_read_csv_data_split_object():
    X, y = read_csv_data(LARGE_DATA_4MB, target_column_index=0, output_dtype="O")
    yX = read_csv_data(LARGE_DATA_4MB, output_dtype="O")
    assert X.shape == (38223, 20)
    assert y.shape == (38223,)
    assert np.array_equal(np.hstack((y.reshape(-1, 1), X)), yX)
    assert X.dtype.kind == "O"
    assert y.dtype.kind == "O"


def test_read_csv_data_split_limited_object():
    total_memory_in_bytes = psutil.virtual_memory().total
    two_mb_in_bytes = _convert_megabytes_to_bytes(2)
    fraction_of_memory_to_use = two_mb_in_bytes / total_memory_in_bytes
    X, y = read_csv_data(
        LARGE_DATA_4MB, target_column_index=0, fit_memory_percent=fraction_of_memory_to_use * 100, output_dtype="O"
    )
    arrays_memory = _get_size_total(X) + _get_size_total(y)
    assert _convert_megabytes_to_bytes(1.9) < arrays_memory <= two_mb_in_bytes
    assert X.dtype.kind == "O"
    assert y.dtype.kind == "O"


@pytest.mark.parametrize("output_dtype", ["O", "U"])
def test_read_to_fit_memory_dangling_element(tmpdir_factory, output_dtype):
    """Test that data is read in correctly when `len(data) = 1 mod batch_size`."""
    data = np.zeros((10, 10)).astype(str)
    for i in range(data.shape[0]):
        data[i, i] = str(i + 1)
    data_dir = tmpdir_factory.mktemp("ten_line_csv")
    data_file = data_dir.join("ten_lines.csv")
    np.savetxt(data_file.strpath, data, delimiter=",", newline="\n", fmt="%s")

    X_read, y_read = _read_to_fit_memory(
        _get_reader(data_dir.strpath, 3),
        psutil.virtual_memory().total,
        output_dtype=output_dtype,
        target_column_index=0,
    )
    assert np.array_equal(data[:, 1:], X_read)
    assert np.array_equal(data[:, 0], y_read)


def test_list_alphabetical():
    """Test for checking 'list_files' returns alphabetically"""
    path = "test/data/csv/mock_datasplitter_output"
    mlio_list_files = list_files(path, pattern="*")
    alphabetical_files = []
    for file in ["excel.csv", "manual.csv", "newline.csv", "oneline.csv"]:
        alphabetical_files.extend(list_files(path + "/" + file, pattern="*"))
    assert mlio_list_files == alphabetical_files


def test_list_recursive():
    """Test for checking 'list_files' lists recursively"""
    assert len(list_files("test/data/csv", pattern="*")) == 10
