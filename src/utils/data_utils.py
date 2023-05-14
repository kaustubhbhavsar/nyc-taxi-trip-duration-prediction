import pandas as pd


def read_data(
    path: str
) -> pd.DataFrame:
    """
    Reads a CSV file from disk and returns its contents as a pandas.DataFrame object.
    
    Parameters:
        path (str): The path to the CSV file to be read.
    
    Returns:
        pd.DataFrame: The contents of the CSV file as a pandas.DataFrame object.
    
    Raises:
        FileNotFoundError: If the specified file does not exist.
        pd.errors.EmptyDataError: If the specified file is empty.
        pd.errors.ParserError: If the CSV file has an invalid format.
    """
    try:
        return pd.read_csv(path)
    except FileNotFoundError:
        print(f"File {path} not found.")
        raise
    except pd.errors.EmptyDataError:
        print(f"File {path} is empty.")
        raise
    except pd.errors.ParserError:
        print(f"File {path} has an invalid format.")
        raise



def write_data(
    df: pd.DataFrame, 
    path: str
) -> None:
    """
    Writes a pandas DataFrame to a CSV file on disk.

    Parameters:
        df (pd.DataFrame): The pandas DataFrame to write to disk.
        path (str): The path to write the CSV file to.

    Raises:
        FileNotFoundError: If the specified file path cannot be found.
    """
    try:
        df.to_csv(path, index=False)
    except FileNotFoundError:
        print(f"File path {path} not found.")
        raise


def concat_dataframes(
    *dfs
) -> pd.DataFrame:
    """
    Concatenates an arbitrary number of dataframes horizontally (i.e., along axis=1).
    
    Args:
        *dfs: variable number of pandas.DataFrame objects to concatenate.
    
    Returns:
        pandas.DataFrame object resulting from the concatenation of the input dataframes.
    """
    return pd.concat(dfs, axis=1)