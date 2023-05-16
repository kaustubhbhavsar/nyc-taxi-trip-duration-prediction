import os
import hopsworks


def login_to_hopsworks(
    project: str,
    api_key: str = None
) -> hopsworks.project:
    """
    Logs in to Hopsworks using the API key stored in a file.

    Parameters:
        project (str): Project name.
        api_key (str): Hopsworks API key.

    Returns:
        hopsworks.project: A Hopsworks project object.

    Raises:
        HopsworksRestAPIError: If unable to connect to hopsworks.
    """
    try:
        # login to hopsworks
        project = hopsworks.login(
            project=project, 
            api_key_value=api_key
        )
        return project
    except hopsworks.exceptions.HopsworksRestAPIError as e:
        print(f"Unable to login to Hopsworks: {e}")
