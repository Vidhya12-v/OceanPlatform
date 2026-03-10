import pandas as pd

# Load dataset
data = pd.read_csv("annual.csv")

# Display first rows
print("First 5 rows:")
print(data.head())

# Display dataset info
print("\nDataset Info:")
print(data.info())

# Check missing values
print("\nMissing Values:")
print(data.isnull().sum())