import pandas as pd
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv("data/annual.csv")

print("Dataset Loaded Successfully")
print(df.head())

# Plot graph
plt.plot(df["Year"], df["Mean"])
plt.xlabel("Year")
plt.ylabel("Temperature")
plt.title("Ocean Temperature Trend")
plt.show()
