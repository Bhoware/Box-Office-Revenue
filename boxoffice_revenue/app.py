from boxoffice_revenue.logger import logging
from boxoffice_revenue.exception import CustomException
from boxoffice_revenue.components.data_ingestion import DataIngestion
from boxoffice_revenue.components.data_ingestion import DataIngestionConfig
from boxoffice_revenue.components.data_transformation import DataTransformationConfig,DataTransformation
#from boxoffice_revenue.components.model_tranier import ModelTrainerConfig,ModelTrainer

import sys


if __name__=="__main__":
    logging.info("The execution has started")

    try:
        data_ingestion_config=DataIngestionConfig()
        data_ingestion=DataIngestion()
        train_data_path,test_data_path=data_ingestion.initiate_data_ingestion()
        print(f"[+] Data Ingestion completed successfully.")
        print(f"    Train data path: {train_data_path}")
        print(f"    Test data path:  {test_data_path}")

        data_transformation_config=DataTransformationConfig()
        data_transformation=DataTransformation()
        train_arr,test_arr,preprocessor_path=data_transformation.initiate_data_transformation(train_data_path,test_data_path)
        print(f"[+] Data Transformation completed successfully.")
        print(f"    Train array shape: {train_arr.shape}")
        print(f"    Test array shape:  {test_arr.shape}")
        print(f"    Preprocessor path: {preprocessor_path}")

        ## Model Training
        # model_trainer=ModelTrainer()
        # print(model_trainer.initiate_model_trainer(train_arr,test_arr))
        
    except Exception as e:
        logging.info("Custom Exception")
        raise CustomException(e,sys)
