import sys
import os
from boxoffice_revenue.exception import CustomException
from boxoffice_revenue.logger import logging
from sklearn.preprocessing import StandardScaler
from dataclasses import dataclass
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from boxoffice_revenue.utils import save_object
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.impute import SimpleImputer
from boxoffice_revenue.utils import save_object


@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path=os.path.join('artifacts','preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.data_transformation_config=DataTransformationConfig()

    def get_data_transformer_object(self):
        try:
            numerical_col = ['budget', 'runtime', 'release_month']
            categorical_col = ['genre', 'language', 'company']
            num_pipeline=Pipeline(
                steps=[
                    ('imputer',SimpleImputer(strategy='median')),
                    ('scaler',StandardScaler())

                ]
            )
            cat_pipeline=Pipeline(
                steps=[
                    ('imputer',SimpleImputer(strategy='most_frequent')),
                    ('onehot',OneHotEncoder(handle_unknown='ignore'))
                ]
            )
            logging.info(f'numerical_column:{numerical_col}')
            logging.info(f'categorical_column:{categorical_col}')

            preprocessor=ColumnTransformer(
                [
                    ('num',num_pipeline,numerical_col),
                    ('cat',cat_pipeline,categorical_col)
                ]
            )
            return preprocessor
        except Exception as e:
            raise CustomException(e,sys)
    def initiate_data_transformation(self,train_data_path,test_data_path):
        try:
            train_df=pd.read_csv(train_data_path)
            test_df=pd.read_csv(test_data_path)
            logging.info("read train and test data")
            logging.info("Obtaining preprocessing object")

            preprocessor=self.get_data_transformer_object()
            target_column = 'revenue'
            numerical_col=['budget','runtime','release_month']
            input_feature_train_df=train_df.drop(columns=[target_column])
            target_feature_train_df=train_df[target_column]

            input_feature_test_df=test_df.drop(columns=[target_column])
            target_feature_test_df=test_df[target_column]

            logging.info("Applying preprocessing object on training and testing data")

            input_feature_train_arr=preprocessor.fit_transform(input_feature_train_df)
            input_feature_test_arr=preprocessor.transform(input_feature_test_df)

            if hasattr(input_feature_train_arr, "toarray"):
                input_feature_train_arr = input_feature_train_arr.toarray()
            if hasattr(input_feature_test_arr, "toarray"):
                input_feature_test_arr = input_feature_test_arr.toarray()

            train_arr=np.c_[input_feature_train_arr,np.array(target_feature_train_df)]
            test_arr=np.c_[input_feature_test_arr,np.array(target_feature_test_df)]

            logging.info(f"saved preprocessing object.")

            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessor
            )
            return(
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path
            )

        except Exception as e:
            raise CustomException(e,sys)
