import os
import pickle
import sys
from math import prod

from sklearn.metrics import r2_score
from sklearn.model_selection import RandomizedSearchCV

from boxoffice_revenue.exception import CustomException
from boxoffice_revenue.logger import logging


def evaluate_models(X_train, y_train, X_test, y_test, models_and_params):
    try:
        report = {}
        fitted = {}

        for name, cfg in models_and_params.items():
            logging.info(f"Tuning {name}...")

            grid_size = prod(len(v) for v in cfg["params"].values())
            search = RandomizedSearchCV(
                cfg["model"],
                cfg["params"],
                n_iter=min(20, grid_size),
                cv=3,
                scoring="r2",
                n_jobs=-1,
                random_state=42,
            )
            search.fit(X_train, y_train)
            best = search.best_estimator_

            train_r2 = r2_score(y_train, best.predict(X_train))
            test_r2 = r2_score(y_test, best.predict(X_test))

            report[name] = test_r2
            fitted[name] = (best, search.best_params_)

            logging.info(f"{name} done. Train R2: {train_r2:.4f}, Test R2: {test_r2:.4f}")

        return report, fitted

    except Exception as e:
        raise CustomException(e, sys)


def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)

def load_object(file_path):
    try:
        with open(file_path, "rb") as file_obj:
            return pickle.load(file_obj)
    except Exception as e:
        raise CustomException(e, sys)