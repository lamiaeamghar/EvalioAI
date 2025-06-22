import pandas as pd
import missingno as msno
import matplotlib.pyplot as plt

df = pd.read_csv(r"C:\Users\hp\Documents\INOCOD\Advanced EvalioIA\data\raw_data.csv")

# 1. Basic Info
print(df.info())
print("\nMissing Values:\n", df.isna().sum())

# 2. Missing Data Visualization
msno.matrix(df)
plt.title("Missing Data Pattern")
plt.show()

# 3. Quick Stats
print("\nDescriptive Stats:\n", df.describe())

# 4. Basic Outliers (for numerical columns)
numeric_cols = df.select_dtypes(include=['number']).columns
df[numeric_cols].boxplot()
plt.xticks(rotation=45)
plt.title("Outlier Detection")
plt.show()
