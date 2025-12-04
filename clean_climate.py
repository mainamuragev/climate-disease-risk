import pandas as pd

# Load the raw NASA POWER CSV (skip metadata)
df = pd.read_csv("kenya_climate.csv", skiprows=15)

# Melt the wide format (JAN–DEC) into long format
df_long = df.melt(
    id_vars=["PARAMETER","YEAR"], 
    value_vars=["JAN","FEB","MAR","APR","MAY","JUN","JUL","AUG","SEP","OCT","NOV","DEC"],
    var_name="MONTH", 
    value_name="VALUE"
)

# Pivot so each parameter becomes a column
df_pivot = df_long.pivot_table(
    index=["YEAR","MONTH"], 
    columns="PARAMETER", 
    values="VALUE"
).reset_index()

# Map month names to numbers
month_map = {
    "JAN":1,"FEB":2,"MAR":3,"APR":4,"MAY":5,"JUN":6,
    "JUL":7,"AUG":8,"SEP":9,"OCT":10,"NOV":11,"DEC":12
}
df_pivot["MONTH_NUM"] = df_pivot["MONTH"].map(month_map)

# Save the cleaned dataset
df_pivot.to_csv("kenya_climate_clean.csv", index=False)

print("✅ Cleaned dataset saved as kenya_climate_clean.csv")
print(df_pivot.head())

