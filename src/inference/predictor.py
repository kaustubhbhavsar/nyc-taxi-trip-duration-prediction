# script used for predicting via the hopsworks deployment
# alternatively, for experimental or personal purposes, we can get model directly from registry or local disk
import joblib
import os

class Predict(object):
    """
    A class for making predictions using a trained XGBoost model.

    Attributes
    ----------
    model : xgboost.Booster
        The trained XGBoost model.

    Methods
    -------
    predict(inputs):
        Predicts the target variable for the given input data.

    """
    def __init__(self):
        """ Download the model artifact and load the same """
        self.model = joblib.load(os.environ["ARTIFACT_FILES_PATH"] + "/final_xgboost.bin")

    def predict(self, inputs):
        """ Serves a prediction request from a trained model """
        return self.model.predict(inputs).tolist()