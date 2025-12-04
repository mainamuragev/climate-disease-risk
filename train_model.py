import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import joblib

# Load the cleaned dataset
df = pd.read_csv("kenya_climate_clean.csv")

# Example: create a dummy target (replace with real labels later)
# For demo purposes, let's say disease risk is high when humidity > 65 and rainfall > 3
df["RiskLabel"] = ((df["RH2M"] > 65) & (df["PRECTOTCORR"] > 3)).astype(int)

# Features and target
X = df[["T2M","RH2M","PRECTOTCORR","ALLSKY_SFC_SW_DWN","WS2M","GWETTOP","GWETROOT"]]
y = df["RiskLabel"]

# Train a simple RandomForest model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)

# Save the model
joblib.dump(model, "model.pkl")

print("✅ Model trained and saved as model.pkl")
