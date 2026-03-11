import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Load dataset
df = pd.read_csv("data/annual.csv")

print("Dataset Loaded for Training")

# Prepare data
X = df[['Year']]
y = df['Mean']

# Train model
model = LinearRegression()
model.fit(X, y)

print("Model Trained Successfully")

# Future prediction years
future_years = pd.DataFrame({'Year':[2025, 2030, 2040, 2050]})

predictions = model.predict(future_years)

print("\nFuture Ocean Temperature Predictions:")
for year, temp in zip(future_years['Year'], predictions):
    print(year, ":", temp)

# Plot predictions
plt.scatter(X, y)
plt.plot(future_years, predictions)
plt.xlabel("Year")
plt.ylabel("Temperature Change")
plt.title("Future Ocean Temperature Prediction")
plt.show()