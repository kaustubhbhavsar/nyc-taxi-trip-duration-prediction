import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from config import config
from pathlib import Path
import pandas as pd
from src.features import engineered_features 
from src.utils import data_utils, hopsworks_utils
import hsfs


def engineer_features(
    df: pd.DataFrame
) -> pd.DataFrame:
    """
    Transforms a pandas.DataFrame object with raw taxi ride data into an engineered feature matrix that can be used for modeling.

    Parameters:
        df (pd.DataFrame): A pandas.DataFrame object that contains raw taxi ride data.

    Returns:
        pd.DataFrame: A pandas.DataFrame object that contains engineered features derived from the raw data.

    Raises:
        TypeError: If df is not a pandas.DataFrame object.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas.DataFrame object.")
    
    try:
        # convert to datetime object
        df['pickup_datetime'] = pd.to_datetime(df.pickup_datetime)

        # separate datetime object into separate columns
        month_df = engineered_features.extract_month(df.pickup_datetime)
        week_df = engineered_features.extract_week(df.pickup_datetime)
        weekday_df = engineered_features.extract_weekday(df.pickup_datetime)
        hour_df = engineered_features.extract_hour(df.pickup_datetime)
        minute_df = engineered_features.extract_minute(df.pickup_datetime)
        minute_of_the_day_df = engineered_features.extract_minute_of_the_day(df.pickup_datetime)

        # get haversine distance as the new feature
        haversine_distance_df = pd.DataFrame(engineered_features.calculate_haversine_distance(
            df['pickup_latitude'].values,
            df['pickup_longitude'].values, 
            df['dropoff_latitude'].values,
            df['dropoff_longitude'].values),
            columns=['haversine_distance_df']
        )

        # get direction as the new feature
        direction_df = pd.DataFrame(engineered_features.calculate_direction(
            df['pickup_latitude'].values,
            df['pickup_longitude'].values, 
            df['dropoff_latitude'].values,
            df['dropoff_longitude'].values),
            columns=['direction_df']
        )

        #  get public holiday and Sunday as new feature
        public_holidays = engineered_features.get_us_federal_holiday_calender(
            min(df.pickup_datetime), 
            max(df.pickup_datetime)
        )
        holiday_df = pd.DataFrame(
            engineered_features.get_holidays(df.pickup_datetime, public_holidays),
            columns=['holiday_df']
        )

        # create feature whereby a value of 1 has been assigne to the hours between 2:00am and 7:00am
        # and a value of 0 has been assigned to all other hours.
        hour_before7_after7_df = pd.DataFrame(
            engineered_features.get_hour_before7_after7(df.pickup_datetime),
            columns=['hour_before7_after7_df']
        )

        # concatenate the generated features into one dataframe
        df_concatenated = data_utils.concat_dataframes(
            df['id'], # id for primary key
            minute_df, 
            minute_of_the_day_df, 
            hour_df, 
            hour_before7_after7_df, 
            weekday_df, 
            week_df, 
            month_df, 
            holiday_df,
            haversine_distance_df, 
            direction_df 
        )
        return df_concatenated
    except Exception as e:
        raise Exception("Error during feature engineering: " + str(e))
    

def update_to_feature_store(
    feature_store: hsfs.feature_store.FeatureStore, 
    feature_group_name: str, 
    df: pd.DataFrame, 
    feature_group_description: str
) -> None:
    """
    Inserts a DataFrame of features to a feature group in a Hopsworks Feature Store, or creates a new feature group if it doesn't exist.

    Args:
        feature_store: The feature store to use for inserting the data.
        feature_group_name: The name of the feature group to insert the data to, or create if it doesn't exist.
        df: A pandas DataFrame containing the features to insert.
        feature_group_description: A description of the feature group to create, if it doesn't exist.

    Returns:
        None

    Raises:
        ValueError: If the feature group name is invalid.
        Exception: If there are any errors while getting, creating, or inserting the feature group.
    """    
     # validate feature group name is alpha-numeric
    if not feature_group_name.isalnum():
        raise ValueError("Feature group name should contain only alphanumeric characters.")
    try:
        # Check if feature group exists
        feature_group = feature_store.get_feature_group(feature_group_name, version=1)
    except:
        # create feature group if it doesn't exist
        try:
            # add event time only in feature group with 'processed' data and not in engineered data
            if feature_group_name=="processed":
                # convert to datetime object
                df['pickup_datetime'] = pd.to_datetime(df.pickup_datetime)
                event_time = "pickup_datetime"
            else:
                event_time = None 
            # create feature group
            feature_group = feature_store.create_feature_group(
                name=feature_group_name,
                version="1",
                description=feature_group_description,
                online_enabled=True,
                primary_key=['id'],
                event_time=event_time
            )
        except Exception as e:
            raise Exception(f"Error creating feature group {feature_group_name}: {e}")
    try:
        # insert features into feature group
        feature_group.insert(
            features=df, 
            write_options={"wait_for_job" : True}
        )
    except Exception as e:
        raise Exception(f"Error inserting features into feature group {feature_group_name}: {e}")



def run(
    project: str
) -> None: 
    """
    Login to the Hopsworks project and write processed and engineered dataframes to the Feature Store.

    Args:
        project (str): The name of the Hopsworks project.

    Returns:
        None

    Raises:
        Exception: If an error occurs while accessing or writing to the Feature Store.
    """
    try:
       # login to hopsworks pass the project as arguement
        project = hopsworks_utils.login_to_hopsworks(project=project)
        # connect to feature store
        fs = project.get_feature_store()
    except Exception as e:
        raise Exception(f"Error connecting to Feature Store in project {project}: {e}")
    
    # read in your data from csv file
    df_processed = data_utils.read_data(Path(config.DATA_DIR, "processed.csv"))
    # call the engineer_features function on your DataFrame
    df_engineered = engineer_features(df_processed.copy())

    # write to feature store -> processed
    fg_name = "processed"
    fg_description= "Processed (raw) dataset after EDA and required processing."
    update_to_feature_store(
        feature_store=fs, 
        feature_group_name=fg_name, 
        df=df_processed, 
        feature_group_description=fg_description
    )
    
    # write to feature store -> engineered
    fg_name="engineered"
    fg_description="Engineered features (dataset) from the processed dataset."
    update_to_feature_store(
        feature_store=fs, 
        feature_group_name=fg_name, 
        df=df_engineered, 
        feature_group_description=fg_description
    )


# run the file
run(project="nyc_taxi_trip_duration")