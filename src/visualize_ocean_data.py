import pandas as pd
import matplotlib.pyplot as plt
import os
import joblib

# Create folder to save plots
os.makedirs("plots", exist_ok=True)

# Load dataset
df = pd.read_csv("data/annual.csv")
print("Dataset Loaded")

# Load trained model for future prediction (optional)
model_path = "models/ocean_model.pkl"
if os.path.exists(model_path):
    model = joblib.load(model_path)
    future_years = pd.DataFrame({'Year': [2025, 2030, 2040, 2050]})
    predictions = model.predict(future_years)
    print("Future Predictions:")
    for year, temp in zip(future_years['Year'], predictions):
        print(f"{year}: {temp:.2f}")

# --------- Graph 1: Temperature Trend ---------
plt.figure(figsize=(10,5))
plt.plot(df["Year"], df["Mean"], marker='o', linestyle='-', color='blue', label="Observed")
plt.title("Global Ocean Temperature Trend")
plt.xlabel("Year")
plt.ylabel("Temperature Anomaly")
plt.grid(True)
plt.legend()
plt.savefig("plots/temperature_trend.png")
plt.show()

# --------- Graph 2: Temperature Distribution ---------
plt.figure(figsize=(8,5))
plt.hist(df["Mean"], bins=30, color='orange', edgecolor='black')
plt.title("Distribution of Ocean Temperature Anomalies")
plt.xlabel("Temperature Anomaly")
plt.ylabel("Frequency")
plt.grid(axis='y')
plt.savefig("plots/temperature_distribution.png")
plt.show()

# --------- Graph 3: Moving Average Trend ---------
df["Moving_Avg"] = df["Mean"].rolling(window=10).mean()
plt.figure(figsize=(10,5))
plt.plot(df["Year"], df["Mean"], label="Original", color='blue')
plt.plot(df["Year"], df["Moving_Avg"], label="10-Year Moving Average", color='red', linewidth=2)
plt.title("Ocean Temperature Trend with Moving Average")
plt.xlabel("Year")
plt.ylabel("Temperature")
plt.grid(True)
plt.legend()
plt.savefig("plots/moving_average_trend.png")
plt.show()

# --------- Optional: Plot Future Predictions ---------
if os.path.exists(model_path):
    plt.figure(figsize=(10,5))
    plt.plot(df["Year"], df["Mean"], label="Observed", color='blue')
    plt.plot(future_years["Year"], predictions, label="Predicted", color='green', marker='x', linestyle='--')
    plt.title("Ocean Temperature: Observed + Future Predictions")
    plt.xlabel("Year")
    plt.ylabel("Temperature")
    plt.grid(True)
    plt.legend()
    plt.savefig("plots/future_predictions.png")
    plt.show()