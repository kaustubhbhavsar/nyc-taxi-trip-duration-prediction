import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..')))

import tempfile
import pytest
import pandas as pd
from src.utils.data_utils import read_data, write_data, concat_dataframes


# generate a temporary file path for testing write_data function
temp_path = os.path.join(tempfile.gettempdir(), "test.csv")


def test_read_data():
    # create a test dataframe and write to disk
    df = pd.DataFrame({"col1": [1, 2, 3], "col2": ["a", "b", "c"]})
    df.to_csv(temp_path, index=False)
    # test read_data function
    read_df = read_data(temp_path)
    assert isinstance(read_df, pd.DataFrame)
    assert read_df.equals(df)


def test_read_data_file_not_found():
    # test read_data function for FileNotFoundError
    with pytest.raises(FileNotFoundError):
        read_data("invalid_path.csv")


def test_write_data():
    # create a test dataframe
    df = pd.DataFrame({"col1": [1, 2, 3], "col2": ["a", "b", "c"]})
    # test write_data function
    write_data(df, temp_path)
    assert os.path.isfile(temp_path)


def test_write_data_file_not_found():
    # create a test dataframe
    df = pd.DataFrame({"col1": [1, 2, 3], "col2": ["a", "b", "c"]})
    # test write_data function for FileNotFoundError
    with pytest.raises(FileNotFoundError):
        write_data(df, "invalid_folder/test.csv")


def test_concat_dataframes():
    # create some sample dataframes to concatenate
    df1 = pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]})
    df2 = pd.DataFrame({"C": [7, 8, 9], "D": [10, 11, 12]})
    # call the function being tested
    result = concat_dataframes(df1, df2)
    # check the output is the expected dataframe
    expected_result = pd.DataFrame(
        {"A": [1, 2, 3], "B": [4, 5, 6], "C": [7, 8, 9], "D": [10, 11, 12]}
    )
    assert result.equals(expected_result)