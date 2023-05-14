import joblib
import os

class Predict(object):
    def __init__(self):
        """ Download the model artifact and load the model """
        self.model = joblib.load(os.environ["ARTIFACT_FILES_PATH"] + "/final_xgboost.bin")

    def predict(self, inputs):
        """ Serves a prediction request from a trained model """
        return self.model.predict(inputs).tolist()