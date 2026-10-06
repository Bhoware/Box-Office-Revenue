import os
import sys

import pandas as pd

from boxoffice_revenue.exception import CustomException
from boxoffice_revenue.logger import logging
from boxoffice_revenue.utils import load_object


class PredictPipeline:
    def __init__(self):
        try:
            self.model = load_object(os.path.join("artifacts", "model.pkl"))
            self.preprocessor = load_object(os.path.join("artifacts", "preprocessor.pkl"))
            logging.info("Model and preprocessor loaded")
        except Exception as e:
            raise CustomException(e, sys)

    def predict(self, features):
        try:
            data_scaled = self.preprocessor.transform(features)
            if hasattr(data_scaled, "toarray"):
                data_scaled = data_scaled.toarray()
            return self.model.predict(data_scaled)
        except Exception as e:
            raise CustomException(e, sys)


class CustomData:
    def __init__(
        self,
        title: str,
        distributor: str,
        MPAA: str,
        genres: str,
        budget: float,
        opening_theaters: float,
        release_days: float,
    ):
        self.title = title
        self.distributor = distributor
        self.MPAA = MPAA
        self.genres = genres
        self.budget = budget
        self.opening_theaters = opening_theaters
        self.release_days = release_days

    def get_data_as_dataframe(self):
        try:
            return pd.DataFrame(
                {
                    "title": [self.title],
                    "distributor": [self.distributor],
                    "MPAA": [self.MPAA],
                    "genres": [self.genres],
                    "budget": [self.budget],
                    "opening_theaters": [self.opening_theaters],
                    "release_days": [self.release_days],
                }
            )
        except Exception as e:
            raise CustomException(e, sys)