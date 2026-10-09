import ast
import os
import sys
from dataclasses import dataclass

import pandas as pd
from sklearn.model_selection import train_test_split

from boxoffice_revenue.exception import CustomException
from boxoffice_revenue.logger import logging


def first_name(s):
    try:
        items = ast.literal_eval(s)
        return items[0]["name"] if items else "Unknown"
    except Exception:
        return "Unknown"


@dataclass
class DataIngestionConfig:
    train_data_path: str = os.path.join('artifacts', 'train.csv')
    test_data_path: str = os.path.join('artifacts', 'test.csv')
    raw_data_path: str = os.path.join('artifacts', 'raw.csv')


class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        try:
            data_path = os.path.join('notebook', 'data', 'boxoffice.csv')
            if not os.path.exists(data_path):
                data_path = os.path.join('boxoffice_revenue', 'notebook', 'data', 'boxoffice.csv')
            raw = pd.read_csv(data_path)
            logging.info("Reading completed")

            df = pd.DataFrame({
                "budget": raw["budget"],
                "runtime": raw["runtime"],
                "release_month": pd.to_datetime(raw["release_date"], errors="coerce").dt.month,
                "genre": raw["genres"].map(first_name),
                "language": raw["original_language"],
                "company": raw["production_companies"].map(first_name),
                "revenue": raw["revenue"],
            })
            df = df[(df["budget"] >= 10000) & (df["revenue"] >= 10000)].dropna()
            top_companies = df["company"].value_counts().head(40).index
            df["company"] = df["company"].where(df["company"].isin(top_companies), "Other")
            logging.info(f"Rows after cleaning: {len(df)}")
            print("Rows after cleaning:", len(df))

            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)
            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)

            train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)
            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)
            logging.info("Data Ingestion is completed")

            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path,
            )
        except Exception as e:
            raise CustomException(e, sys)


if __name__ == "__main__":
    from boxoffice_revenue.components.data_transformation import DataTransformation
    from boxoffice_revenue.components.model_trainer import ModelTrainer

    train_path, test_path = DataIngestion().initiate_data_ingestion()
    train_arr, test_arr, _ = DataTransformation().initiate_data_transformation(train_path, test_path)
    print("R2:", ModelTrainer().initiate_model_trainer(train_arr, test_arr))