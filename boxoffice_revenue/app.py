import uvicorn
from fastapi import FastAPI, Form, Request          
from fastapi.responses import HTMLResponse          
from fastapi.templating import Jinja2Templates      
from pydantic import BaseModel                      
from boxoffice_revenue.logger import logging
from boxoffice_revenue.utils import CustomException
from boxoffice_revenue.pipeline.prediction_pipeline import CustomData, PredictPipeline  

app = FastAPI(title="Box Office Revenue Predictor") 
templates = Jinja2Templates(directory="templates")   
pipeline = PredictPipeline()                         


class MovieInput(BaseModel):    
    title: str                  
    distributor: str            
    MPAA: str                   
    genres: str                 
    budget: float               
    opening_theaters: float     
    release_days: float         


@app.get("/", response_class=HTMLResponse)   
def index(request: Request):                 
    return templates.TemplateResponse(request, "index.html", {"result": None})


@app.post("/", response_class=HTMLResponse)  
def predict_form(
    request: Request,                        
    title: str = Form(...),                  
    distributor: str = Form(...),            
    MPAA: str = Form(...),
    genres: str = Form(...),
    budget: float = Form(...),               
    opening_theaters: float = Form(...),
    release_days: float = Form(...),
):
    data = CustomData(title, distributor, MPAA, genres, budget, opening_theaters, release_days)
    pred = pipeline.predict(data.get_data_as_dataframe())
    return templates.TemplateResponse(request, "index.html", {"result": f"${pred[0]:,.0f}"})


@app.post("/predict")                        
def predict_api(movie: MovieInput):          
    data = CustomData(**movie.model_dump()) 
    pred = pipeline.predict(data.get_data_as_dataframe()) 
    return {"predicted_domestic_revenue": float(pred[0])} 
    