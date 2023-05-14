import pandas as pd
import numpy as np
import pytest
from math import isclose
from src.features import engineered_features


@pytest.fixture
def datetime():
    return pd.Series([
        "2022-01-01 00:30:00",
        "2022-01-02 06:00:00",
        "2022-01-03 12:45:00",
        "2022-01-04 18:12:00",
    ])


def test_extract_month(datetime):
    datetime = pd.to_datetime(datetime)
    expected_output = pd.DataFrame({
        "month": [1, 1, 1, 1]
    })
    assert engineered_features.extract_month(datetime).equals(expected_output)


def test_extract_week(datetime):
    datetime = pd.to_datetime(datetime)
    expected_output = pd.DataFrame({
        "week": [52, 52, 1, 1]
    })
    assert engineered_features.extract_week(datetime).equals(expected_output)


def test_extract_weekday(datetime):
    datetime = pd.to_datetime(datetime)
    expected_output = pd.DataFrame({
        "weekday": [5, 6, 0, 1]
    })
    assert engineered_features.extract_weekday(datetime).equals(expected_output)


def test_extract_hour(datetime):
    datetime = pd.to_datetime(datetime)
    expected_output = pd.DataFrame({
        "hour": [0, 6, 12, 18]
    })
    assert engineered_features.extract_hour(datetime).equals(expected_output)


def test_extract_minute(datetime):
    datetime = pd.to_datetime(datetime)
    expected_output = pd.DataFrame({
        "minute": [30, 0, 45, 12]
    })
    assert engineered_features.extract_minute(datetime).equals(expected_output)


def test_extract_minute_of_the_day(datetime):
    datetime = pd.to_datetime(datetime)
    expected_output = pd.DataFrame({
        "minute_of_the_day": [30, 360, 765, 1092]
    })
    assert engineered_features.extract_minute_of_the_day(datetime).equals(expected_output)


@pytest.fixture
def holiday_list():
    return ["2022-01-01", "2022-01-02"]

def test_get_holidays(datetime, holiday_list):
    datetime = pd.to_datetime(datetime)
    expected_output = np.array([0, 1, 0, 0])
    result = engineered_features.get_holidays(datetime, holiday_list)
    assert len(result) == len(datetime)
    assert np.array_equal(result, expected_output)


def test_get_us_federal_holiday_calender():
    expected_output = pd.DataFrame({
        "date": [
            pd.to_datetime("2016-01-01"),
            pd.to_datetime("2016-01-18"),
            pd.to_datetime("2016-02-15")
        ]
    })
    output = engineered_features.get_us_federal_holiday_calender(
        pd.to_datetime("2016-01-01 00:00:00"),
        pd.to_datetime("2016-02-20 00:00:00")
    )
    assert output.equals(expected_output)


def test_get_hour_before7_after7(datetime):
    datetime = pd.to_datetime(datetime)
    output = engineered_features.get_hour_before7_after7(datetime)
    expected_output = np.array([1, 0, 1, 1])
    assert np.array_equal(output, expected_output)


def test_calculate_haversine_distance():
    pickup_latitude = 37.7749
    pickup_longitude = -122.4194
    dropoff_latitude = 40.7128
    dropoff_longitude = -74.0060
    expected_distance = 4127.41  # expected distance in km
    distance = engineered_features.calculate_haversine_distance(
        pickup_latitude, pickup_longitude, dropoff_latitude, dropoff_longitude
    )
    assert isclose(distance, expected_distance, rel_tol=1e-2)


@pytest.mark.parametrize(
    "pickup_latitude, pickup_longitude, dropoff_latitude, dropoff_longitude, expected_direction",
    [
        (0, 0, 1, 1, 45),
        (52.2297, 21.0122, 52.4064, 16.9252, 303),
        (-33.859972, 151.211111, -33.863526, 151.202741, 255),
        (37.7749, -122.4194, 40.7128, -74.0060, 71),
    ],
)
def test_calculate_direction(
    pickup_latitude, pickup_longitude, dropoff_latitude, dropoff_longitude, expected_direction
):
    direction = engineered_features.calculate_direction(
        pickup_latitude, pickup_longitude, dropoff_latitude, dropoff_longitude
    )
    assert np.isclose(direction, expected_direction, rtol=0.1)