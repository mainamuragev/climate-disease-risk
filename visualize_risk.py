import pandas as pd
import matplotlib.pyplot as plt

# Load predictions dataset
df = pd.read_csv("kenya_disease_risk.csv")

# Summarize average disease risk per year
yearly_summary = df.groupby("YEAR")["Risk_Diseased"].mean().reset_index()
print("\n📊 Average Disease Risk per Year:")
print(yearly_summary)

# Loop through all years and save a plot for each
for year in df["YEAR"].unique():
    subset = df[df["YEAR"] == year]
    plt.figure(figsize=(10,6))
    plt.plot(subset["MONTH_NUM"], subset["Risk_Diseased"], marker="o", color="red")
    plt.xticks(subset["MONTH_NUM"], subset["MONTH"])
    plt.title(f"Disease Risk Trend ({year})")
    plt.xlabel("Month")
    plt.ylabel("Risk of Disease")
    plt.grid(True)
    plt.savefig(f"risk_trend_{year}.png")
    plt.close()
    print(f"✅ Saved risk_trend_{year}.png")

# Combined multi-year plot
plt.figure(figsize=(12,7))
for year in df["YEAR"].unique():
    subset = df[df["YEAR"] == year]
    plt.plot(subset["MONTH_NUM"], subset["Risk_Diseased"], marker="o", label=str(year))

plt.xticks(range(1,13), ["JAN","FEB","MAR","APR","MAY","JUN","JUL","AUG","SEP","OCT","NOV","DEC"])
plt.title("Disease Risk Trends Across Years")
plt.xlabel("Month")
plt.ylabel("Risk of Disease")
plt.legend(loc="upper left", bbox_to_anchor=(1,1))
plt.grid(True)
plt.tight_layout()
plt.savefig("risk_trends_all_years.png")
print("✅ Combined visualization saved as risk_trends_all_years.png")

