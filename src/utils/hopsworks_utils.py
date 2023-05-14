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
            api_key_file=r'E:\NYC Taxi Trip Duration Prediction\config\hopsworks_api_key'
            )
        return project
    except hopsworks.exceptions.HopsworksRestAPIError as e:
        print(f"Unable to login to Hopsworks: {e}")