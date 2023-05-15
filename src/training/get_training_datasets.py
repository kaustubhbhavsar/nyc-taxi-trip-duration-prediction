import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from config import config
from pathlib import Path
import pandas as pd
from src.utils import data_utils, hopsworks_utils
from hsfs.feature_view import FeatureView


def get_training_data(
    feature_view: FeatureView, 
    training_dataset_version: int
) -> pd.DataFrame:
    """
    Retrieves the training data from the specified version of a feature view and returns it as a tuple.

    Args:
        feature_view (FeatureView): The feature view to retrieve the training data from.
        training_dataset_version (int): The version of the training dataset to retrieve.

    Returns:
        pd.DataFrame: Data in the form of a Pandas DataFrame.
    """
    features_df, _ = feature_view.get_training_data(training_dataset_version=training_dataset_version)
    return features_df


def run(
    project: str
) -> None: 
    """
    Retrieves the feature view instance from the feature store and extracts training and testing data.
  
    Parameters:
        project (str): The name of the Hopsworks project to connect to.

    Returns:
        None.

    Raises:
        Exception: If there is an error connecting to the Hopsworks feature store or 
        retrieving featureviews.
    """
    try:
       # login to hopsworks pass the project as arguement
        project = hopsworks_utils.login_to_hopsworks(project=project)
        # connect to feature store
        fs = project.get_feature_store()
    except Exception as e:
        raise Exception(f"Error connecting to Feature Store in project {project}: {e}")

    try:
        # get feature view instance
        feature_view = fs.get_feature_view(
            name='combined_features',
            version=1
        )
    except Exception as e:
        raise Exception(f"Error retrieving feature view: {e}")

    # get training data
    train_df = get_training_data(feature_view=feature_view, training_dataset_version=1)

    # get testing data
    test_df = get_training_data(feature_view=feature_view, training_dataset_version=2)

    # write train and test datasets to file
    if not os.path.exists(Path(config.DATA_DIR, "training_datasets")):
        os.makedirs(Path(config.DATA_DIR, "training_datasets")) # make dir if it doesn't exist
    data_utils.write_data(train_df, Path(config.DATA_DIR, "training_datasets", "train.csv"))
    data_utils.write_data(test_df, Path(config.DATA_DIR, "training_datasets", "test.csv"))


# run the file
run(project="nyc_taxi_trip_duration")