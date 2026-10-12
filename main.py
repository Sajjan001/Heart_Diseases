import sys
from typing import Annotated

import joblib
import pandas as pd
from fastapi import FastAPI, Form, Path, Request
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field
from sklearn._loss import _loss as sklearn_loss

sys.modules.setdefault("_loss", sklearn_loss)
model_data = joblib.load("heart.joblib")
model = model_data["model"]
features = model_data["features"]


app = FastAPI()
templates = Jinja2Templates(directory="templates")

class heartData(BaseModel):
    age: int = Field(...,ge = 0,le =120, description="Age of the patient")
    sex: int = Field(..., description="Sex")
    cp: int = Field(le = 3 ,ge = 0, description="Chest pain type")
    trestbps: int = Field(..., description="Resting blood pressure")
    chol: int = Field(..., description="Serum cholesterol")
    fbs: int = Field(..., description="Fasting blood sugar")
    restecg: int = Field(..., description="Resting electrocardiographic results")
    thalach: int = Field(..., description="Maximum heart rate achieved")
    exang: int = Field(..., description="Exercise-induced angina")
    oldpeak: float = Field(..., description="ST depression induced by exercise")
    slope: int = Field(..., description="Slope of the peak exercise ST segment")
    ca: int = Field(..., description="Number of major vessels colored by fluoroscopy")
    thal: int = Field(..., description="Thalassemia")

@app.get("/")
def home(request: Request):
    return templates.TemplateResponse(request=request, name="index.html")


@app.post("/predict")
def predict(request: Request, data:Annotated[heartData, Form()]):
    input_data = pd.DataFrame([data.model_dump()])
    input_data = input_data[features]
    prediction = model.predict(input_data)
    if prediction[0] == 1:
        result = "The patient is likely to have heart disease."
    else:
        result = "The patient is unlikely to have heart disease."

    return templates.TemplateResponse(request=request, name="index.html", context={"result": result})