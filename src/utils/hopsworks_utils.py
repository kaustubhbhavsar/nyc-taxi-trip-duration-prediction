import os
import hopsworks


def login_to_hopsworks(
    project: str,
    streamlit_api_key: str = None
) -> hopsworks.project:
    """
    Logs in to Hopsworks using the API key stored in a file.

    Parameters:
        project (str): Project name.
        streamlit_api_key (str): Hopsworks API key from Streamlit secrets.

    Returns:
        hopsworks.project: A Hopsworks project object.

    Raises:
        HopsworksRestAPIError: If unable to connect to hopsworks.
    """
    try:
        # assigining 'hopsworks key' to 'api_key_value'
        if streamlit_api_key is None:
            api_key_value = os.environ.get('HOPSWORKS_API_KEY') # accessing api key from secrets
        else:
            api_key_value = streamlit_api_key
        # login to hopsworks
        project = hopsworks.login(
            project=project, 
            api_key_value=api_key_value
        )
        return project
    except hopsworks.exceptions.HopsworksRestAPIError as e:
        print(f"Unable to login to Hopsworks: {e}")
