from pathlib import Path

import uvicorn
from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

from boxoffice_revenue.logger import logging
from boxoffice_revenue.pipeline.prediction_pipeline import CustomData, PredictPipeline

app = FastAPI(title="Box Office Revenue Predictor")
templates_dir = Path(__file__).resolve().parent / "templates"
templates = Jinja2Templates(directory=str(templates_dir))
pipeline = PredictPipeline()


class MovieInput(BaseModel):
    budget: float
    runtime: float
    release_month: int
    genre: str
    language: str
    company: str


@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    return templates.TemplateResponse(request, "index.html", {"result": None})


@app.post("/", response_class=HTMLResponse)
def predict_form(
    request: Request,
    budget: float = Form(...),
    runtime: float = Form(...),
    release_month: int = Form(...),
    genre: str = Form(...),
    language: str = Form(...),
    company: str = Form(...),
):
    data = CustomData(budget, runtime, release_month, genre, language, company)
    pred = pipeline.predict(data.get_data_as_dataframe())
    return templates.TemplateResponse(request, "index.html", {"result": f"${pred[0]:,.0f}"})


@app.post("/predict")
def predict_api(movie: MovieInput):
    data = CustomData(**movie.model_dump())
    pred = pipeline.predict(data.get_data_as_dataframe())
    return {"predicted_domestic_revenue": float(pred[0])}