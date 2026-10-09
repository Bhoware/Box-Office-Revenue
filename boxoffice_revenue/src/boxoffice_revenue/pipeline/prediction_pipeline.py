import os
import sys

from pathlib import Path

import pandas as pd

from boxoffice_revenue.exception import CustomException
from boxoffice_revenue.logger import logging
from boxoffice_revenue.utils import load_object


class PredictPipeline:
    def __init__(self, model_path: str | None = None, preprocessor_path: str | None = None):
        try:
            self.model_path = model_path or self._resolve_artifact("model.pkl")
            self.preprocessor_path = preprocessor_path or self._resolve_artifact("preprocessor.pkl")
            self.model = load_object(self.model_path)
            self.preprocessor = load_object(self.preprocessor_path)
            logging.info("Model and preprocessor loaded")
        except Exception as e:
            raise CustomException(e, sys)

    @staticmethod
    def _resolve_artifact(filename: str) -> str:
        cwd_path = os.path.join("artifacts", filename)
        if os.path.exists(cwd_path):
            return cwd_path
        sub_path = os.path.join("boxoffice_revenue", "artifacts", filename)
        if os.path.exists(sub_path):
            return sub_path
        pkg_path = Path(__file__).resolve().parents[3] / "artifacts" / filename
        if pkg_path.exists():
            return str(pkg_path)
        return cwd_path

    def predict(self, features):
        try:
            data_scaled = self.preprocessor.transform(features)
            if hasattr(data_scaled, "toarray"):
                data_scaled = data_scaled.toarray()
            return self.model.predict(data_scaled)
        except Exception as e:
            raise CustomException(e, sys)


class CustomData:
    def __init__(self, budget: float, runtime: float, release_month: int,
                 genre: str, language: str, company: str):
        self.budget = budget
        self.runtime = runtime
        self.release_month = release_month
        self.genre = genre
        self.language = language
        self.company = company

    def get_data_as_dataframe(self):
        try:
            return pd.DataFrame({
                "budget": [self.budget],
                "runtime": [self.runtime],
                "release_month": [self.release_month],
                "genre": [self.genre],
                "language": [self.language],
                "company": [self.company],
            })
        except Exception as e:
            raise CustomException(e, sys)