from boxoffice_revenue.pipeline.prediction_pipeline import CustomData, PredictPipeline

data = CustomData(
    title="Inception",
    distributor="Warner Bros.",
    MPAA="PG-13",
    genres="Drama",
    budget=160000000,
    opening_theaters=3500,
    release_days=60,
)
print(PredictPipeline().predict(data.get_data_as_dataframe()))