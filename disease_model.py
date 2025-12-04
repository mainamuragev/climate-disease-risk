import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# Load dataset
df = pd.read_csv("plant_disease_dataset.csv")

# Features and target
X = df[['temperature','humidity','rainfall','soil_pH']]
y = df['disease_present']

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))

# Climate change scenario (+2°C, +10% humidity)
X_future = X_test.copy()
X_future['temperature'] += 2
X_future['humidity'] *= 1.10

current_risks = model.predict_proba(X_test)[:,1]
future_risks = model.predict_proba(X_future)[:,1]

print("Mean current risk:", np.mean(current_risks))
print("Mean future risk:", np.mean(future_risks))

