import os
import sys
from pathlib import Path

# Ensure boxoffice_revenue root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from fastapi.testclient import TestClient
from app import app
from boxoffice_revenue.pipeline.prediction_pipeline import CustomData, PredictPipeline
from boxoffice_revenue.utils import load_object

client = TestClient(app)


def test_prediction_pipeline():
    data = CustomData(
        budget=160000000,
        runtime=148,
        release_month=7,
        genre="Action",
        language="en",
        company="Universal Pictures",
    )

    preprocessor_path = os.path.join(PROJECT_ROOT, "artifacts", "preprocessor.pkl")
    model_path = os.path.join(PROJECT_ROOT, "artifacts", "model.pkl")

    preprocessor = load_object(file_path=preprocessor_path)
    model = load_object(file_path=model_path)

    data_scaled = preprocessor.transform(data.get_data_as_dataframe())
    predicted_value = model.predict(data_scaled)
    assert predicted_value is not None
    assert len(predicted_value) == 1
    assert predicted_value[0] > 0


def test_predict():
    payload = {
        "budget": 160000000,
        "runtime": 148,
        "release_month": 7,
        "genre": "Action",
        "language": "en",
        "company": "Universal Pictures",
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "predicted_domestic_revenue" in data
    assert data["predicted_domestic_revenue"] > 0


def test_predictions_vary():
    small = {"budget": 2000000, "runtime": 95, "release_month": 10,
             "genre": "Drama", "language": "en", "company": "Other"}
    big = {"budget": 150000000, "runtime": 124, "release_month": 6,
           "genre": "Action", "language": "en", "company": "Universal Pictures"}
    p_small = client.post("/predict", json=small).json()["predicted_domestic_revenue"]
    p_big = client.post("/predict", json=big).json()["predicted_domestic_revenue"]
    assert p_big > 3 * p_small


if __name__ == "__main__":
    test_prediction_pipeline()
    test_predict()
    test_predictions_vary()
    print("All tests passed!")
