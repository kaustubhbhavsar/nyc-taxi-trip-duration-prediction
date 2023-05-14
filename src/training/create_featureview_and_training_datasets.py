import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

import datetime
from typing import Tuple
from src.utils import hopsworks_utils
from hsfs.feature_group import FeatureGroup
from hsfs.feature_store import FeatureStore
from hsfs.feature_view import FeatureView


def get_feature_groups(
    fs: FeatureStore
) -> Tuple[FeatureGroup, FeatureGroup]:
    """
    Retrieve the processed and engineered feature groups from a given FeatureStore.

    Parameters:
        fs (FeatureStore): A FeatureStore instance to connect to.

    Returns:
        A tuple containing the FeatureGroup objects for the processed and engineered groups, respectively.

    Raises:
        Exception: If an error occurs while retrieving the feature groups.
    """
    try:
        fg_processed = fs.get_feature_group(name="processed", version=1)
        fg_engineered = fs.get_feature_group(name="engineered", version=1)
        return fg_processed, fg_engineered
    except Exception as e:
        raise Exception(f"Error retrieving feature groups: {e}")


def select_features(
    fg_processed: FeatureGroup, 
    fg_engineered: FeatureGroup
) -> FeatureGroup:
    """
    Joins the processed and engineered feature groups on the "id" column and returns
    the resulting FeatureGroup.

    Args:
        fg_processed (FeatureGroup): The processed feature group.
        fg_engineered (FeatureGroup): The engineered feature group.

    Returns:
        FeatureGroup: The joined feature group.

    Raises:
        Exception: If an error occurs while selecting features.
    """
    try:
        fg_query = fg_processed.select_all().join(fg_engineered.select_all(), on=["id"])
        return fg_query
    except Exception as e:
        raise Exception(f"Error selecting features: {e}")


def create_feature_view(
    fs: FeatureStore, 
    fg_query: FeatureGroup
) -> FeatureView:
    """
    Creates a new feature view in the given FeatureStore or retrieves an existing one with the 
    given name and version.
    
    Args:
        fs (FeatureStore): The FeatureStore to use for creating or retrieving the FeatureView.
        fg_query (FeatureGroup): The FeatureGroup containing the features to use for the FeatureView.
    
    Returns:
        FeatureView: The newly created or retrieved FeatureView.
    
    Raises:
        Exception: If there is an error creating the FeatureView.
    """
    try:
        feature_view = fs.create_feature_view(
            name="combined_features",
            query=fg_query,
            description="Dataset combined from engineered and processed feature group.",
            version=1
        )
        return feature_view
    except Exception as e:
        raise Exception(f"Error creating feature view: {e}")


def create_training_dataset(
    feature_view: FeatureView,
    start_time: str,
    end_time: str,
    description: str
) -> Tuple:
    """
    Creates a training dataset based on a specified time range from the provided FeatureView.

    Args:
        feature_view (FeatureView): A FeatureView object containing the data to use for training.
        start_time (str): The start time of the time range for the training dataset, in the format "%Y-%m-%d %H:%M:%S".
        end_time (str): The end time of the time range for the training dataset, in the format "%Y-%m-%d %H:%M:%S".
        description (str): A description for the training dataset.

    Returns:
        (td_version, job): A tuple of the training dataset version and the job object.
    """
    # convert time of type string to datetime format
    date_format = "%Y-%m-%d %H:%M:%S" # datetime format
    start_time = datetime.datetime.strptime(start_time, date_format)
    end_time = datetime.datetime.strptime(end_time, date_format)
    version, job = feature_view.create_training_data(
        start_time=start_time,
        end_time=end_time,
        description=description,
        data_format="csv",
        write_options={"wait_for_job": True}
    )
    return version, job
    

def run(
    project: str
) -> None: 
    """
    Connects to a Hopsworks project's feature store, selects features from two feature groups,
    creates a feature view, splits the data into train and test sets based on time.

    Parameters:
        project (str): The name of the Hopsworks project to connect to.

    Returns:
        None.

    Raises:
        Exception: If there is an error connecting to the Hopsworks feature store.
    """
    try:
       # login to hopsworks pass the project as arguement
        project = hopsworks_utils.login_to_hopsworks(project=project)
        # connect to feature store
        fs = project.get_feature_store()
    except Exception as e:
        raise Exception(f"Error connecting to Feature Store in project {project}: {e}")

    # get feature group with processed and engineered dataset
    fg_processed, fg_engineered = get_feature_groups(fs)

    # select features using query from both the feature groups
    fg_query = select_features(fg_processed, fg_engineered)

    # get or create feature view
    feature_view = create_feature_view(fs, fg_query)

    # create train data
    job, version = create_training_dataset(
        feature_view=feature_view,
        start_time="2016-01-01 00:00:17",
        end_time="2016-05-31 23:59:59",
        description="Creating training dataset from 2016/Jan/01 to 2016/May/31."
    
    )
    
    # create test data
    job, version = create_training_dataset(
        feature_view=feature_view,
        start_time="2016-06-01 00:00:00",
        end_time="2016-06-30 23:59:39",
        description="Creating testing dataset from 2016/June/01 to 2016/June/30."
    )


# run the file
run(project="nyc_taxi_trip_duration")