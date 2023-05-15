import os
import hopsworks


def login_to_hopsworks(
    project: str
) -> hopsworks.project:
    """
    Logs in to Hopsworks using the API key stored in a file.

    Parameters:
        project (str): Project name.

    Returns:
        hopsworks.project: A Hopsworks project object.

    Raises:
        HopsworksRestAPIError: If unable to connect to hopsworks.
    """
    try:
        project = hopsworks.login(
            project=project, 
            api_key_value=os.environ.get('HOPSWORKS_API_KEY') # accessing api key from secrets
        )
        return project
    except hopsworks.exceptions.HopsworksRestAPIError as e:
        print(f"Unable to login to Hopsworks: {e}")
