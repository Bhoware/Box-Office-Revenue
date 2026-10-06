from fastapi import FastAPI, Form, Request          # FastAPI = the app itself; Form = read HTML form fields; Request = the raw incoming request
from fastapi.responses import HTMLResponse          # tells FastAPI a route returns an HTML page, not JSON
from fastapi.templating import Jinja2Templates      # renders HTML files that contain {{ placeholders }}
from pydantic import BaseModel                      # base class for defining and validating JSON input
from boxoffice_revenue.logger import logging
from boxoffice_revenue.utils import CustomException
from boxoffice_revenue.pipeline.prediction_pipeline import CustomData, PredictPipeline  # your own prediction code

app = FastAPI(title="Box Office Revenue Predictor")  # creates the app; title appears on the /docs page
templates = Jinja2Templates(directory="templates")   # look for HTML files in the "templates" folder
pipeline = PredictPipeline()                         # runs ONCE at startup: loads model.pkl and preprocessor.pkl into memory


class MovieInput(BaseModel):    # describes what the JSON body of POST /predict must look like
    title: str                  # must be text
    distributor: str            # must be text
    MPAA: str                   # must be text
    genres: str                 # must be text
    budget: float               # must be a number (a string like "abc" is rejected with a 422 error)
    opening_theaters: float     # must be a number
    release_days: float         # must be a number


@app.get("/", response_class=HTMLResponse)   # when a browser opens http://127.0.0.1:8000/ (GET), run the function below
def index(request: Request):                 # FastAPI passes in the request automatically
    # render index.html; result=None means "don't show a prediction yet"
    return templates.TemplateResponse(request, "index.html", {"result": None})


@app.post("/", response_class=HTMLResponse)  # when the HTML form is submitted (POST to the same URL), run this instead
def predict_form(
    request: Request,                        # needed by the template renderer
    title: str = Form(...),                  # read "title" from the form; "..." means required
    distributor: str = Form(...),            # the names must match the name="..." attributes in index.html
    MPAA: str = Form(...),
    genres: str = Form(...),
    budget: float = Form(...),               # form values arrive as text; FastAPI converts them to float
    opening_theaters: float = Form(...),
    release_days: float = Form(...),
):
    # pack the values into your CustomData container (order matches its __init__)
    data = CustomData(title, distributor, MPAA, genres, budget, opening_theaters, release_days)
    # data -> one-row DataFrame -> preprocessor -> model -> array of predictions
    pred = pipeline.predict(data.get_data_as_dataframe())
    # pred[0] is the single prediction; :,.0f formats it like $149,542,776
    return templates.TemplateResponse(request, "index.html", {"result": f"${pred[0]:,.0f}"})


@app.post("/predict")                        # JSON endpoint: other programs call this, not a browser form
def predict_api(movie: MovieInput):          # FastAPI reads the JSON body and validates it against MovieInput
    data = CustomData(**movie.model_dump())  # model_dump() turns the object into a dict; ** unpacks it into keyword arguments
    pred = pipeline.predict(data.get_data_as_dataframe())  # same prediction steps as above
    return {"predicted_domestic_revenue": float(pred[0])}  # a dict is automatically sent back as JSON; float() converts the numpy number
    