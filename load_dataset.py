import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("annual.csv")

print("Dataset Loaded Successfully")
print(df.head())

# Plot temperature trend
plt.figure(figsize=(10,5))
plt.plot(df["Year"], df["Mean"])
plt.title("Global Ocean Temperature Trend")
plt.xlabel("Year")
plt.ylabel("Temperature Anomaly")
plt.show()
