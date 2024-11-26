import sys
import os
import numpy as np
import pandas as pd
from src.exception import CustomException
from src.utils import load_object
from src.logger import logging


class PredictPipeline:
    def __init__(self):
        self.model_path = os.path.join("artifacts", "model.pkl")

    def preprocess_for_prediction(self, data: pd.DataFrame):
        """
        Preprocess the data to create sequences of 30 lagged values for prediction.
        """
        try:
            logging.info("Preprocessing data for prediction.")
            if len(data) < 30:
                raise CustomException("Insufficient data: At least 30 records are required for prediction.")

            sequences = []
            for i in range(30, len(data)):
                sequences.append(data.iloc[i-30:i, 0])  # Create sequences of 30 lagged values
            
            logging.info("Data preprocessed into sequences.")
            return np.array(sequences)

        except Exception as e:
            raise CustomException(e, sys)

    def predict(self, data: pd.DataFrame):
        """
        Predict using the pre-trained model.
        """
        try:
            logging.info("Loading the trained model for prediction.")
            model = load_object(file_path=self.model_path)

            logging.info("Preprocessing data for prediction.")
            data_sequences = self.preprocess_for_prediction(data)

            logging.info("Making predictions.")
            predictions = model.predict(data_sequences)

            return predictions

        except Exception as e:
            raise CustomException(e, sys)


class CustomData:
    """
    Handles raw input data for prediction.
    """
    def __init__(self, sms_counts: list):
        """
        Args:
            sms_counts (list): A list of SMS counts (numeric values).
        """
        self.sms_counts = sms_counts

    def get_data_as_data_frame(self):
        """
        Convert the list of SMS counts into a DataFrame.

        Returns:
            pd.DataFrame: DataFrame containing the SMS counts.
        """
        try:
            logging.info("Converting SMS count data to DataFrame.")
            return pd.DataFrame({"sms_count": self.sms_counts})

        except Exception as e:
            raise CustomException(e, sys)
