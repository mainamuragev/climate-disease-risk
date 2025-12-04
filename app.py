from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib

# Load model once at startup
model = joblib.load("model.pkl")

app = FastAPI(title="Plant Disease Risk API")

# Define input schema
class PredictionInput(BaseModel):
    year: int
    month: int
    county: str
    soil_ph: float | None = None  # optional

@app.post("/predict")
def predict(input: PredictionInput):
    # For demo: load climate data from CSV
    df = pd.read_csv("kenya_climate_clean.csv")

    # Filter by year + month
    subset = df[(df["YEAR"] == input.year) & (df["MONTH_NUM"] == input.month)]

    if subset.empty:
        return {"error": "No climate data found for that year/month"}

    # Features
    X = subset[["T2M","RH2M","PRECTOTCORR","ALLSKY_SFC_SW_DWN","WS2M","GWETTOP","GWETROOT"]]

    # Predict
    probs = model.predict_proba(X)
    risk_score = float(probs[:,1].mean())

    # Risk level
    if risk_score < 0.3:
        level = "Low"
    elif risk_score < 0.6:
        level = "Medium"
    else:
        level = "High"

    return {
        "year": input.year,
        "month": input.month,
        "county": input.county,
        "risk_score": round(risk_score, 2),
        "risk_level": level
    }
