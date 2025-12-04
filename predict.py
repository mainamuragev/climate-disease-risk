import pandas as pd
import joblib

# Load the cleaned climate dataset
df = pd.read_csv("kenya_climate_clean.csv")

# Select the features for prediction
X = df[["T2M","RH2M","PRECTOTCORR","ALLSKY_SFC_SW_DWN","WS2M","GWETTOP","GWETROOT"]]

# Load your trained model
model = joblib.load("model.pkl")

# Predict probabilities
probs = model.predict_proba(X)

# Attach predictions to the dataframe
df["Risk_Healthy"] = probs[:,0]
df["Risk_Diseased"] = probs[:,1]

# Save predictions to a new CSV
df.to_csv("kenya_disease_risk.csv", index=False)

print("✅ Predictions saved as kenya_disease_risk.csv")
print(df[["YEAR","MONTH","MONTH_NUM","T2M","RH2M","PRECTOTCORR","Risk_Diseased"]].head())

