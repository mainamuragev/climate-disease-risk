from fastapi import FastAPI
import pandas as pd
import joblib

app = FastAPI()

# Health check
@app.get("/health")
def health():
    return {"status": "ok"}

# Root welcome route
@app.get("/")
def root():
    return {"message": "Welcome to the Climate Disease Risk API"}

# Prediction route using your trained model
@app.get("/predict")
def predict(location: str = "Kenya", year: int = 2025):
    # Load model (make sure model.pkl is in project root)
    model = joblib.load("model.pkl")
    # Example: use year as a simple feature
    prediction = model.predict([[year]])[0]
    return {"location": location, "year": year, "predicted_risk": float(prediction)}

# Route to preview climate dataset
@app.get("/data")
def get_data(rows: int = 5):
    df = pd.read_csv("kenya_climate_clean.csv")
    return df.head(rows).to_dict(orient="records")

