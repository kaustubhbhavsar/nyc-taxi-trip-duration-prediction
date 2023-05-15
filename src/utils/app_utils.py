import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))


from config import config
import hopsworks
import joblib
from pathlib import Path
import pandas as pd
import shutil
from src.features import engineered_features
from src.utils import data_utils
from typing import Any


def process_input(
    vendor_id: int,
    passenger_count: int,
    pickup_latitude: float, 
    pickup_longitude: float, 
    dropoff_latitude: float, 
    dropoff_longitude: float,
    pickup_datetime:float
) -> pd.DataFrame:
    """
    The function takes in several input parameters and generates features from them. 
    The function returns a Pandas DataFrame containing the engineered features.

    Args:
        vendor_id (int): The ID of the taxi vendor. 
        passenger_count (int): The number of passengers in the taxi ride. 
        pickup_latitude (float): The latitude of the pickup location. 
        pickup_longitude (float): The longitude of the pickup location. 
        dropoff_latitude (float): The latitude of the dropoff location. 
        dropoff_longitude (float): The longitude of the dropoff location. 
        pickup_datetime (float): The timestamp of the pickup time in seconds since the epoch. 
    
    Returns:
        df_concatenated (Pandas DataFrame): A DataFrame containing the engineered features as columns.
    """
    # create a DataFrame from the dictionary created from the inputs
    df_input = pd.DataFrame.from_dict({
        'vendor_id': [vendor_id],
        'passenger_count': [passenger_count],
        'pickup_latitude': [pickup_latitude],
        'pickup_longitude': [pickup_longitude],
        'dropoff_latitude': [dropoff_latitude],
        'dropoff_longitude': [dropoff_longitude],
        'pickup_datetime': [pd.to_datetime(pickup_datetime, unit='s').strftime('%Y-%m-%d %H:%M:%S')]
    })

    # separate datetime object into separate columns
    pickup_dt_object = pd.to_datetime(df_input.pickup_datetime) # convert to datetime object
    month = engineered_features.extract_month(pickup_dt_object)
    week = engineered_features.extract_week(pickup_dt_object)
    weekday = engineered_features.extract_weekday(pickup_dt_object)
    minute_of_the_day = engineered_features.extract_minute_of_the_day(pickup_dt_object)

    # get haversine distance as the new feature
    haversine_distance = pd.DataFrame(engineered_features.calculate_haversine_distance(
        df_input['pickup_latitude'].values,
        df_input['pickup_longitude'].values, 
        df_input['dropoff_latitude'].values,
        df_input['dropoff_longitude'].values),
        columns=['haversine_distance']
    )

    # get direction as the new feature
    direction = pd.DataFrame(engineered_features.calculate_direction(
        df_input['pickup_latitude'].values,
        df_input['pickup_longitude'].values, 
        df_input['dropoff_latitude'].values,
        df_input['dropoff_longitude'].values),
        columns=['direction']
    )

    #  get public holiday and Sunday as new feature
    public_holidays = engineered_features.get_us_federal_holiday_calender(
        min(pickup_dt_object), 
        max(pickup_dt_object)
    )
    holiday = pd.DataFrame(
        engineered_features.get_holidays(pickup_dt_object, public_holidays),
        columns=['holiday']
    )

    # create feature whereby a value of 1 has been assigne to the hours between 2:00am and 7:00am
    # and a value of 0 has been assigned to all other hours.
    h2AMto7AM_7AMto2AM = pd.DataFrame(
        engineered_features.get_hour_before7_after7(pickup_dt_object),
        columns=['h2amto7am_7amto2am']
    )

    # concatenate the generated features into one dataframe
    df_concatenated = data_utils.concat_dataframes(
        h2AMto7AM_7AMto2AM,
        direction,
        df_input['dropoff_latitude'],
        df_input['dropoff_longitude'],
        haversine_distance,
        holiday,
        minute_of_the_day,
        month,
        df_input['passenger_count'],
        df_input['pickup_latitude'],
        df_input['pickup_longitude'],
        df_input['vendor_id'],
        week,
        weekday
    )
   
    return df_concatenated


def get_model(
    project: hopsworks.project, 
    model_name: str, 
    version: int
) -> Any:
    """
    This function loads a trained model from Hopsworks. If the model already exists in the specified 
    directory, it will be loaded from that directory. Otherwise, the model will be downloaded from 
    the Hopsworks Model Registry, and then saved to the specified directory before being loaded.

    Args:
        project (hopsworks.project): The Hopsworks project object.
        model_name (str): The name of the trained model in the Hopsworks Model Registry.
        version (int): The version number of the trained model.

    Returns:
        The trained model object.
    """
    if os.path.exists(Path(config.MODEL_DIR, "1", "final_xgb_model.bin")):
        pass
    else:
        # get model registry
        mr = project.get_model_registry()
        # download model
        model = mr.get_model(model_name, version)
        model_dir = model.download()
        # move the downloaded model directory to the model directory
        shutil.move(model_dir, Path(config.MODEL_DIR))
    # load and return model
    model = joblib.load(Path(config.MODEL_DIR, "1" ,"final_xgb_model.bin"))
    return model