import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

import hsfs
import pandas as pd
from typing import Tuple


def get_df(
    feature_store: hsfs.feature_store.FeatureStore, 
    feature_view_name: str,
    feature_view_version: int,
    training_dataset_version: int
) -> pd.DataFrame:
    """
    This function takes as input a feature store instance, the name and version of a feature view, 
    and the version of a training dataset. It returns a Pandas DataFrame with the data 
    for the given feature view and version.

    Parameters:
        feature_store: A hsfs.feature_store.FeatureStore instance representing the feature store 
            where the feature view and training dataset are located.
        feature_view_name: A string with the name of the feature view to retrieve the data from.
        feature_view_version: An integer with the version of the feature view to retrieve the data from.
        training_dataset_version: An integer with the version of the training dataset to retrieve the data from.

    Returns:
        A Pandas DataFrame containing the data for the specified feature view and version.
    """
    # get feature view instance 
    fv = feature_store.get_feature_view(
        name=feature_view_name,
        version=feature_view_version
    )
    # get data as dataframe
    df, _ = fv.get_training_data(training_dataset_version=training_dataset_version)
    # return dataframe
    return df


def get_specific_features(
    test_df: pd.DataFrame, 
    prediction_df: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame, list]:
    """
    Returns two dataframes: one containing the specific features required for prediction, 
    and another containing the same specific features, but for the test dataset.
    
    Args:
        test_df (pd.DataFrame): The test dataset.
        prediction_df (pd.DataFrame): The predictions dataset.
        
    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: A tuple of two dataframes. The first dataframe contains 
        the specific features required for prediction, and the second dataframe contains the same 
        specific features, but for the test dataset.
    """
    # there is mismatch in feature names of test_df and prediction_df
    # rename the columns of test_df using a dictionary
    test_df = test_df.rename(
        columns={
            'direction_df': 'direction',
            'trip_duration': 'prediction'
        }
    )

    # selecting specific columns
    selected_features = ['direction', 'dropoff_latitude', 'dropoff_longitude', 'minute_of_the_day', 
        'pickup_latitude', 'pickup_longitude', 'prediction']
    prediction_df = prediction_df.loc[:, selected_features]
    test_df = test_df.loc[:, selected_features]

    return test_df.to_numpy(), prediction_df.to_numpy(), selected_features
