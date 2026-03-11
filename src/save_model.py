import pandas as pd
from sklearn.linear_model import LinearRegression
import joblib
import os

# Absolute path to CSV
df = pd.read_csv(r"C:\OceanPlatform\data\annual.csv")

# Train model
X = df[['Year']]
y = df['Mean']
model = LinearRegression()
model.fit(X, y)

# Correct path inside your project root
models_path = os.path.join("models")
os.makedirs(models_path, exist_ok=True)

# Save model
joblib.dump(model, os.path.join(models_path, "ocean_model.pkl"))
print("Model saved at:", os.path.join(models_path, "ocean_model.pkl"))