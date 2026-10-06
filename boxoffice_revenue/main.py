import sys

from boxoffice_revenue.components.data_ingestion import DataIngestion
from boxoffice_revenue.components.data_transformation import DataTransformation
from boxoffice_revenue.components.model_trainer import ModelTrainer
from boxoffice_revenue.exception import CustomException
from boxoffice_revenue.logger import logging

if __name__ == "__main__":
    logging.info("The execution has started")

    try:
        train_data_path, test_data_path = DataIngestion().initiate_data_ingestion()
        print("[+] Data Ingestion completed successfully.")
        print(f"    Train data path: {train_data_path}")
        print(f"    Test data path:  {test_data_path}")

        train_arr, test_arr, preprocessor_path = (
            DataTransformation().initiate_data_transformation(
                train_data_path, test_data_path
            )
        )
        print("[+] Data Transformation completed successfully.")
        print(f"    Train array shape: {train_arr.shape}")
        print(f"    Test array shape:  {test_arr.shape}")
        print(f"    Preprocessor path: {preprocessor_path}")

        print("[+] Model Training result (best test R2):")
        print(ModelTrainer().initiate_model_trainer(train_arr, test_arr))

    except Exception as e:
        logging.info("Custom Exception")
        raise CustomException(e, sys)