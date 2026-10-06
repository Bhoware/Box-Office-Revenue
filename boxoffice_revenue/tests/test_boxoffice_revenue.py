from boxoffice_revenue.pipeline.prediction_pipeline import CustomData, PredictPipeline
from boxoffice_revenue.utils import load_object
data = CustomData(
    title="Inception",
    distributor="Warner Bros.",
    MPAA="PG-13",
    genres="Drama",
    budget=160000000,
    opening_theaters=3500,
    release_days=60,
)

preprocessor_path='artifacts/preprocessor.pkl'
model_path='artifacts/model.pkl'

preprocessor = load_object(file_path=preprocessor_path)
model = load_object(file_path=model_path)


data_scaled = preprocessor.transform(data.get_data_as_dataframe())
predicted_value = model.predict(data_scaled)
print(predicted_value)