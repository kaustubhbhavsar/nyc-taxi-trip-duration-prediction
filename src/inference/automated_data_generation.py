"""
This file is a utility script to automate the task of generating random taxi trip data, predicting 
trip duration on that random data using the trained model in production, and uploading the entire data
to feature store. 
This is only a way to generate lot of data, specifically for use in monitoring section.
This file is not a part of the main project files, and not required if random data generation isn't required. 
This file is run on scheduled using Github Actions.

To launch a streamlit web app for on-demand predictions, 
please refer to the 'app.py' file located in the current directory.
"""
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

import pandas as pd
import random
from src.utils import app_utils, hopsworks_utils
import time
from typing import Tuple


def generate_random_coordinates(
    min_lat: float, 
    max_lat: float, 
    min_lon: float, 
    max_lon: float
) -> Tuple[float, float]:
    """
    Generates random latitude and longitude coordinates within specified bounds.

    Args:
        min_lat (float): The minimum latitude value for the region.
        max_lat (float): The maximum latitude value for the region.
        min_lon (float): The minimum longitude value for the region.
        max_lon (float): The maximum longitude value for the region.

    Returns:
        tuple: A tuple containing the randomly generated latitude and longitude coordinates.
    """
    latitude = random.uniform(min_lat, max_lat)
    longitude = random.uniform(min_lon, max_lon)
    return latitude, longitude


def generate_trip_data() -> Tuple[int, int, float, float, float, float, float]:
    """
    Generates random taxi trip data.

    Args:
        None

    Returns:
        Tuple[int, int, float, float, float, float, float]: A tuple containing vendor_id, passenger_count,
        pickup_latitude, pickup_longitude, dropoff_latitude, dropoff_longitude, and pickup_datetime.
    """
    # generate 'vendor_id' with more weightage given to 'vendor_id=2'
    # higher weightage is as per the observations from EDA that 'vendor_id=2' has most number of records
    vendor_id = random.choices([1, 2], weights=[1, 3], k=1)[0]

    # generate 'passenger_count' with equal probability
    passenger_count = random.randint(1, 6)

    # generate current timestamp for 'pickup_datetime'
    pickup_datetime = time.time()

    # generate random pickup locations
    pickup_latitude, pickup_longitude = generate_random_coordinates(
        min_lat=40.5, 
        max_lat=41.0, 
        min_lon=-74.25, 
        max_lon=-73.50
    )

    # generate random dropoff locations
    dropoff_latitude, dropoff_longitude = generate_random_coordinates(
        min_lat=40.5, 
        max_lat=41.2, 
        min_lon=-74.5, 
        max_lon=-73.0
    )

    return (
        vendor_id, passenger_count, pickup_latitude,
        pickup_longitude, dropoff_latitude, dropoff_longitude,
        pickup_datetime
    )

def run(
    project: str
) -> None: 
    """
    Generates random taxi trip data and inserts it into the feature store.

    Args:
        project (str): The name of the project.

    Returns:
        None.
    """
    # get random taxi trip data
    vendor_id, passenger_count, pickup_latitude, \
    pickup_longitude, dropoff_latitude, dropoff_longitude, \
    pickup_datetime = generate_trip_data()


    # get all features engineered
    df_predictor = app_utils.process_input(
        vendor_id,
        passenger_count,
        pickup_latitude, 
        pickup_longitude, 
        dropoff_latitude, 
        dropoff_longitude,
        pickup_datetime
    )

    # login to hopsworks pass the project as arguement
    project = hopsworks_utils.login_to_hopsworks(project="nyc_taxi_trip_duration")
    
    # get model
    model = app_utils.get_model(project=project, model_name="final_xgboost", version=1)
        
    # make predictions
    prediction = model.predict(df_predictor) # in seconds
    df_prediction = pd.DataFrame.from_dict({ # create DataFrame from a prediction
        'prediction': prediction
    })

    # connect to feature store
    try:
        feature_store = project.get_feature_store() 
    except Exception as e:
        raise Exception(f"Error connecting to Feature Store at Hopsworks {project}: {e}")

    # get feature group
    feature_group_name = "predictions"
    prediction_feature_group = feature_store.get_feature_group(
        name=feature_group_name, 
        version=1
    )
       
    # create DataFrame from a pickup_datetime
    df_pickup_datetime = pd.DataFrame.from_dict({
        'pickup_datetime': [pd.to_datetime(pickup_datetime, unit='s').strftime('%Y-%m-%d %H:%M:%S')]
    })
          
    # concat dataframes
    df_updated = pd.concat([df_pickup_datetime, df_predictor, df_prediction], axis=1)
        
    # to make feature datatypes compatible with that of feature store
    # warning: this is unnecessary storage of very small number to big size dtype, 
    # and should be resolved
    df_updated = df_updated.astype({
        'h2amto7am_7amto2am': 'int64', 
        'holiday': 'int64'
    })

    # insert to feature group
    prediction_feature_group.insert(df_updated)


# run the file
run(project="nyc_taxi_trip_duration")