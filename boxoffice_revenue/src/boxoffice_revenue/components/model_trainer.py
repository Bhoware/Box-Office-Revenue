import os
import sys
from dataclasses import dataclass

import numpy as np
from sklearn.compose import TransformedTargetRegressor
from sklearn.dummy import DummyRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from xgboost import XGBRegressor

from boxoffice_revenue.exception import CustomException
from boxoffice_revenue.logger import logging
from boxoffice_revenue.utils import save_object


@dataclass
class ModelTrainerConfig:
    train_model_file_path: str = os.path.join("artifacts", "model.pkl")


class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_arr, test_arr):
        try:
            X_train, y_train = train_arr[:, :-1], train_arr[:, -1]
            X_test, y_test = test_arr[:, :-1], test_arr[:, -1]

            model = TransformedTargetRegressor(
                regressor=XGBRegressor(
                    n_estimators=400, learning_rate=0.03, max_depth=3,
                    min_child_weight=5, subsample=0.8, colsample_bytree=0.8,
                    reg_lambda=5.0, random_state=42, n_jobs=-1,
                ),
                func=np.log1p, inverse_func=np.expm1,
            )
            baseline = TransformedTargetRegressor(
                regressor=DummyRegressor(strategy="mean"),
                func=np.log1p, inverse_func=np.expm1,
            )
            model.fit(X_train, y_train)
            baseline.fit(X_train, y_train)

            pred = model.predict(X_test)
            r2 = r2_score(y_test, pred)
            mae = mean_absolute_error(y_test, pred)
            base_mae = mean_absolute_error(y_test, baseline.predict(X_test))

            print(f"Test R2: {r2:.3f} | MAE: {mae:,.0f} | baseline MAE: {base_mae:,.0f} "
                  f"| pred std: {np.std(pred):,.0f} | true std: {np.std(y_test):,.0f}")
            logging.info(f"R2={r2:.4f} MAE={mae:.0f} baseline_MAE={base_mae:.0f}")

            if r2 < 0.1 or mae > 0.95 * base_mae:
                print("WARNING: model is no better than predicting the average. "
                      "The features probably don't explain revenue in this dataset.")

            save_object(file_path=self.model_trainer_config.train_model_file_path, obj=model)
            return r2

        except Exception as e:
            raise CustomException(e, sys)