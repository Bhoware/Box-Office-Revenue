import os
import sys
from dataclasses import dataclass


import numpy as np
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from boxoffice_revenue.exception import CustomException
from boxoffice_revenue.logger import logging
from boxoffice_revenue.utils import evaluate_models, save_object


@dataclass
class ModelTrainerConfig:
    train_model_file_path: str = os.path.join("artifacts", "model.pkl")


class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_arr, test_arr):
        try:
            logging.info("Started model training")
            X_train, y_train = train_arr[:, :-1], train_arr[:, -1]
            X_test, y_test = test_arr[:, :-1], test_arr[:, -1]

            models_and_params = {
                "GradientBoosting": {
                    "model": GradientBoostingRegressor(random_state=42),
                    "params": {
                        "n_estimators": [100, 200, 300, 500],
                        "learning_rate": [0.01, 0.05, 0.1, 0.2],
                        "max_depth": [3, 5, 7, 9],
                        "min_samples_split": [2, 5, 10],
                        "min_samples_leaf": [1, 2, 4],
                        "subsample": [0.7, 0.8, 0.9, 1.0],
                    },
                },
                "AdaBoost": {
                    "model": AdaBoostRegressor(random_state=42),
                    "params": {
                        "n_estimators": [50, 100, 200, 300, 500],
                        "learning_rate": [0.01, 0.05, 0.1, 0.5, 1.0],
                        "loss": ["linear", "square", "exponential"],
                    },
                },
                "XGBoost": {
                    "model": XGBRegressor(random_state=42, verbosity=0),
                    "params": {
                        "n_estimators": [100, 200, 300, 500],
                        "learning_rate": [0.01, 0.05, 0.1, 0.2],
                        "max_depth": [3, 5, 7, 9],
                        "min_child_weight": [1, 3, 5],
                        "subsample": [0.7, 0.8, 0.9, 1.0],
                        "colsample_bytree": [0.7, 0.8, 0.9, 1.0],
                        "reg_alpha": [0, 0.01, 0.1, 1],
                        "reg_lambda": [0.5, 1, 1.5, 2],
                    },
                },
                "LightGBM": {
                    "model": LGBMRegressor(random_state=42, verbose=-1),
                    "params": {
                        "n_estimators": [100, 200, 300, 500],
                        "learning_rate": [0.01, 0.05, 0.1, 0.2],
                        "max_depth": [3, 5, 7, 10, -1],
                        "num_leaves": [15, 31, 63, 127],
                        "min_child_samples": [5, 10, 20, 30],
                        "subsample": [0.7, 0.8, 0.9, 1.0],
                        "colsample_bytree": [0.7, 0.8, 0.9, 1.0],
                        "reg_alpha": [0, 0.01, 0.1, 1],
                        "reg_lambda": [0, 0.01, 0.1, 1],
                    },
                },
                "RandomForest": {
                    "model": RandomForestRegressor(random_state=42),
                    "params": {
                        "n_estimators": [100, 200, 300, 500],
                        "max_depth": [3, 5, 7, 9],
                        "min_samples_split": [2, 5, 10],
                        "min_samples_leaf": [1, 2, 4],
                        "max_features": ["sqrt", "log2", None],
                    },
                },
                "DecisionTree": {
                    "model": DecisionTreeRegressor(random_state=42),
                    "params": {
                        "max_depth": [3, 5, 7, 9],
                        "min_samples_split": [2, 5, 10],
                        "min_samples_leaf": [1, 2, 4],
                    },
                },
                "LinearRegression": {
                    "model": LinearRegression(),
                    "params": {"fit_intercept": [True, False]},
                },
                "CatBoost": {
                    "model": CatBoostRegressor(
                        random_state=42, verbose=0, allow_writing_files=False
                    ),
                    "params": {
                        "iterations": [100, 200, 300, 500],
                        "learning_rate": [0.01, 0.05, 0.1, 0.2],
                        "depth": [3, 5, 7, 9],
                        "l2_leaf_reg": [1, 3, 5, 7],
                    },
                },
            }

            model_report, fitted = evaluate_models(
                X_train, y_train, X_test, y_test, models_and_params
            )

            best_model_name = max(model_report, key=model_report.get)
            best_model_score = model_report[best_model_name]
            best_model, best_params = fitted[best_model_name]

            logging.info(f"All model scores: {model_report}")
            logging.info(f"Best model: {best_model_name}, R2: {best_model_score}")

            print("Model scores (test R2):", model_report)
            if best_model_score < 0.6:
                logging.warning(f"Best R2 {best_model_score:.4f} is below 0.6, saving anyway")

            save_object(
                file_path=self.model_trainer_config.train_model_file_path,
                obj=best_model,
            )

            predicted = best_model.predict(X_test)
            r2 = r2_score(y_test, predicted)
            mae = mean_absolute_error(y_test, predicted)
            rmse = np.sqrt(mean_squared_error(y_test, predicted))

           

            return r2

        except Exception as e:
            raise CustomException(e, sys)