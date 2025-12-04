import pandas as pd
import numpy as np

# Set random seed for reproducibility
np.random.seed(42)

# Number of samples (increase if you want more data)
n_samples = 10000

# Generate synthetic farm IDs and regions
farm_ids = np.arange(1, n_samples+1)
regions = np.random.choice(["North", "South", "East", "West"], size=n_samples)

# Generate environmental features
data = {
    "farm_id": farm_ids,
    "region": regions,
    "temperature": np.random.uniform(15, 35, n_samples),   # °C
    "humidity": np.random.uniform(40, 100, n_samples),     # %
    "rainfall": np.random.uniform(0, 20, n_samples),       # mm
    "soil_pH": np.random.uniform(4.5, 7.5, n_samples),     # pH
}

df = pd.DataFrame(data)

# Simple rule to simulate disease presence
df["disease_present"] = (
    (df["temperature"] > 25) & (df["humidity"] > 70) & (df["rainfall"] > 5)
).astype(int)

# Save to CSV
df.to_csv("plant_disease_dataset.csv", index=False)

print("✅ Large dummy dataset created: plant_disease_dataset.csv")
print(df.head())
print("Total samples:", len(df))
