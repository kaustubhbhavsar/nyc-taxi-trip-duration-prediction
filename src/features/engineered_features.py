import numpy as np
import pandas as pd
from pandas.tseries.holiday import (
    USFederalHolidayCalendar,  # to find public holidays in US
)


def extract_month(
    datetime: pd.Series
) -> pd.DataFrame:
    """
    Extracts the month from a pandas datetime column.

    Args:
        datetime (pandas.Series): Column of datetime values from which to extract the month.

    Returns:
        pandas.DataFrame: 
            DataFrame containing a single column "month", with the extracted month
            values for each element in the input datetime.
    """
    dic = {"month": datetime.dt.month}
    return pd.DataFrame(data=dic)


def extract_week(
    datetime: pd.Series
) -> pd.DataFrame:
    """
    Extracts the week from a pandas datetime column.

    Args:
        datetime (pandas.Series): Column of datetime values from which to extract the week.

    Returns:
        pandas.DataFrame: 
            DataFrame containing a single column "week", with the extracted week 
            values for each element in the input datetime.
    """
    dic = {"week": datetime.dt.week}
    return pd.DataFrame(data=dic)


def extract_weekday(
    datetime: pd.Series
) -> pd.DataFrame:
    """
    Extracts the weekday from a pandas datetime column.

    Args:
        datetime (pandas.Series): Column of datetime values from which to extract the weekday.

    Returns:
        pandas.DataFrame: 
            DataFrame containing a single column "weekday", with the extracted weekday 
            values for each element in the input datetime.
    """
    dic = {"weekday": datetime.dt.weekday}
    return pd.DataFrame(data=dic)


def extract_hour(
    datetime: pd.Series
) -> pd.DataFrame:
    """
    Extracts the hour from a pandas datetime column.

    Args:
        datetime (pandas.Series): Column of datetime values from which to extract the hour.

    Returns:
        pandas.DataFrame: D
            ataFrame containing a single column "hour", with the extracted hour values for 
            each element in the input datetime.
    """
    dic = {"hour": datetime.dt.hour}
    return pd.DataFrame(data=dic)


def extract_minute(
    datetime: pd.Series
) -> pd.DataFrame:
    """
    Extracts the minute from a pandas datetime column.

    Args:
        datetime (pandas.Series): Column of datetime values from which to extract the minute.

    Returns:
        pandas.DataFrame: 
            DataFrame containing a single column "minute", with the extracted minute values 
            for each element in the input datetime.
    """
    dic = {"minute": datetime.dt.minute}
    return pd.DataFrame(data=dic)


def extract_minute_of_the_day(
    datetime: pd.Series
) -> pd.DataFrame:
    """
    Extracts the 'minute of the day' from a pandas datetime column.

    Args:
        datetime (pandas.Series): 
            Column of datetime values from which to extract the 'minute of the day'.

    Returns:
        pandas.DataFrame: 
            DataFrame containing a single column "minute_of_the_day", with the extracted 
            'minute of the day' values for each element in the input datetime.
    """
    dic = {"minute_of_the_day": datetime.dt.hour * 60 + datetime.dt.minute}
    return pd.DataFrame(data=dic)


def get_holidays(
    datetime: pd.Series, 
    holiday_list: list
) -> np.ndarray:
    """
    Returns a binary indicator for whether each element in a pandas datetime column 
    is a holiday (public holiday and sunday) or a weekday.

    Args:
        datetime (pandas.Series): Column of datetime values for which to compute the holiday indicator.
        holiday_list (list): List of dates corresponding to public holidays. 

    Returns:
        numpy.ndarray: 
            1D array containing a binary indicator for whether each element in the input datetime 
            is a holiday (public holiday and sunday) or weekday. A value of 1 indicates that the 
            corresponding element is a holiday (public holiday and sunday), and a value of 0 indicates 
            that it is not.
    """
    return np.where(
        (datetime.dt.weekday == 6) | (datetime.dt.date.isin(holiday_list)), 1, 0
    )


def get_us_federal_holiday_calender(
    datetime_min: pd.Timestamp, 
    datetime_max: pd.Timestamp
) -> pd.DataFrame:
    """
    Return a DataFrame of US Federal holidays between two dates.

    Args:
        datetime_min (pandas.Timestamp): Start date for holiday search (inclusive).
        datetime_max (pandas.Timestamp): End date for holiday search (inclusive).

    Returns:
        pandas.DataFrame: DataFrame with column 'date' containing US Federal holidays.
    """
    calender = USFederalHolidayCalendar()
    return pd.DataFrame(calender.holidays(start=datetime_min, end=datetime_max), columns=["date"])


def get_hour_before7_after7(
    datetime: pd.Series
) -> np.ndarray:
    """
    Returns a binary indicator for whether each element in a pandas datetime column corresponds to a 
    time between 2:00 PM and 7:00 AM or between 7:00 AM and 2 PM.

    Args:
        datetime (pandas.Series):
            Column of datetime values for which to compute the hour indicator.

    Returns:
        numpy.ndarray: 
            1D array containing a binary indicator for whether each element in the input datetime
            corresponds to a time between 2:00 PM and 7:00 AM or between 7:00 AM and 2 PM. 
            A value of 1 indicates that the corresponding element is between 2:00 PM and 7:00 AM, 
            and a value of 0 indicates that it is between 7:00 AM and 2 PM.
    """
    return np.where(
        (datetime.dt.hour == 0) | (datetime.dt.hour == 1) | (datetime.dt.hour > 7), 1, 0
    )


def calculate_haversine_distance(
    pickup_latitude: float, 
    pickup_longitude: float, 
    dropoff_latitude: float, 
    dropoff_longitude: float,
) -> float:
    """
    Calculate the haversine distance (in km) between two points on the earth's surface, given their
    latitude and longitude coordinates.

    Args:
        pickup_latitude (float): The latitude coordinate of the pickup location.
        pickup_longitude (float): The longitude coordinate of the pickup location.
        dropoff_latitude (float): The latitude coordinate of the dropoff location.
        dropoff_longitude (float): The longitude coordinate of the dropoff location.

    Returns:
        float: 
            The haversine distance (in km) between the two points specified by their 
            latitude and longitude coordinates.
    """
    pickup_latitude, pickup_longitude, dropoff_latitude, dropoff_longitude = map(
        np.radians, (pickup_latitude, pickup_longitude, dropoff_latitude, dropoff_longitude)
    )
    AVG_EARTH_RADIUS = 6371  # in km
    lat = dropoff_latitude - pickup_latitude
    lng = dropoff_longitude - pickup_longitude
    d = (
        np.sin(lat * 0.5) ** 2
        + np.cos(pickup_latitude) * np.cos(dropoff_latitude) * np.sin(lng * 0.5) ** 2
    )
    h = 2 * AVG_EARTH_RADIUS * np.arcsin(np.sqrt(d))
    return h


def calculate_direction(
    pickup_latitude: float, 
    pickup_longitude: float, 
    dropoff_latitude: float, 
    dropoff_longitude: float,
) -> float:
    """
    Calculate the direction (in degrees) from the pickup location to the dropoff location 
    on the earth's surface, given their latitude and longitude coordinates.

    Args:
        pickup_latitude (float): The latitude coordinate of the pickup location.
        pickup_longitude (float): The longitude coordinate of the pickup location.
        dropoff_latitude (float): The latitude coordinate of the dropoff location.
        dropoff_longitude (float): The longitude coordinate of the dropoff location.

    Returns:
        float: The direction (in degrees) from the pickup location to the dropoff location.
    """
    pickup_latitude, pickup_longitude, dropoff_latitude, dropoff_longitude = map(
        np.radians, (pickup_latitude, pickup_longitude, dropoff_latitude, dropoff_longitude)
    )
    AVG_EARTH_RADIUS = 6371  # in km
    lng_delta_rad = dropoff_longitude - pickup_longitude
    y_coordinate = np.sin(lng_delta_rad) * np.cos(dropoff_latitude)
    x_coordinate = np.cos(pickup_latitude) * np.sin(dropoff_latitude) - np.sin(pickup_latitude) * np.cos(dropoff_latitude) * np.cos(lng_delta_rad)
    return (np.degrees(np.arctan2(y_coordinate, x_coordinate)) + 360) % 360