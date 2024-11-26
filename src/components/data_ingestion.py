import os
import sys
from src.exception import CustomException
from sklearn.preprocessing import MinMaxScaler
from src.logger import logging
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass

@dataclass
class DataIngestionConfig:
    train_data_path: str = os.path.join('artifacts', "train.csv")
    test_data_path: str = os.path.join('artifacts', "test.csv")
    raw_data_path: str = os.path.join('artifacts', "data.csv")

class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def preprocess_data(self, df):
        """
        Perform data preprocessing: sort, remove columns, filter outliers, normalize, and sequence the data.
        """
        try:
            logging.info("Preprocessing data...")
            df['dc_date'] = pd.to_datetime(df['dc_date'])
            df = df.set_index('dc_date').sort_values(by='dc_date')

            # Drop unnecessary columns
            df = df.drop(columns=['company_name', 'Unnamed: 0'], errors='ignore')

            # Remove outliers using IQR
            Q1 = df['dc_daily_count'].quantile(0.25)
            Q3 = df['dc_daily_count'].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            df = df[(df['dc_daily_count'] >= lower_bound) & (df['dc_daily_count'] <= upper_bound)]

            # Normalize the data
            scaler = MinMaxScaler()
            df['dc_daily_count'] = scaler.fit_transform(df[['dc_daily_count']])

            # Create sequences for regression modeling
            Dataset = []
            target = []
            for i in range(30, len(df)):
                Dataset.append(df.iloc[i-30:i, 0])
                target.append(df.iloc[i, 0])

            # Convert to arrays
            Dataset = np.array(Dataset)
            target = np.array(target)

            logging.info("Preprocessing completed successfully.")
            return Dataset, target
        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_ingestion(self):
        logging.info("Entered the data ingestion method or component")
        try:
            df = pd.read_csv('notebook/data/Cleaned Basic daily.csv')
            logging.info('Read the dataset as dataframe')

            Dataset, target = self.preprocess_data(df)

            # Split into training and testing sets
            split_ratio = 0.9
            split_idx = int(split_ratio * len(Dataset))
            X_train, X_test = Dataset[:split_idx], Dataset[split_idx:]
            y_train, y_test = target[:split_idx], target[split_idx:]

            # Save data as artifacts
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)
            pd.DataFrame({'data': list(X_train), 'target': y_train}).to_csv(
                self.ingestion_config.train_data_path, index=False, header=True
            )
            pd.DataFrame({'data': list(X_test), 'target': y_test}).to_csv(
                self.ingestion_config.test_data_path, index=False, header=True
            )

            logging.info("Data ingestion completed successfully.")
            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path,
            )
        except Exception as e:
            raise CustomException(e, sys)
