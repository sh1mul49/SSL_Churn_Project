import sys
import numpy as np
import pandas as pd
from dataclasses import dataclass
from src.exception import CustomException
from src.logger import logging

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts', "preprocessor.pkl")

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def initiate_data_transformation(self, train_path, test_path):
        try:
            logging.info("Reading train and test data for transformation...")
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            # Extract data and target from the train and test sets
            X_train = np.array(list(train_df['data'].apply(eval)))
            y_train = train_df['target'].values
            X_test = np.array(list(test_df['data'].apply(eval)))
            y_test = test_df['target'].values

            logging.info("Data transformation completed successfully.")
            return X_train, y_train, X_test, y_test
        except Exception as e:
            raise CustomException(e, sys)
