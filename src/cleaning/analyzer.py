import pandas as pd
import missingno as msno
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# 1. Load data with error handling
try:
    df = pd.read_csv(r"C:\Users\hp\Documents\INOCOD\Advanced EvalioIA\data\data_cleaned_no_outliers.csv")
    print("Data loaded successfully!")
except Exception as e:
    print(f"Error loading data: {e}")
    exit()

# 2. Enhanced Data Quality Check
def enhanced_quality_report(df):
    print("=== ENHANCED DATA QUALITY REPORT ===")
    print(f"\nShape: {df.shape} (rows, columns)")
    
    # Missing values analysis
    missing = df.isna().sum()
    missing_pct = (missing / len(df)) * 100
    missing_report = pd.DataFrame({
        'Missing Values': missing,
        'Percentage (%)': missing_pct.round(2)
    })
    print("\nMissing Values Report:")
    print(missing_report[missing_report['Missing Values'] > 0].sort_values('Percentage (%)', ascending=False))
    
    # Data types analysis
    print("\nData Types Summary:")
    print(df.dtypes.value_counts())
    
    # Numeric vs categorical summary
    numeric_cols = df.select_dtypes(include=np.number).columns
    cat_cols = df.select_dtypes(exclude=np.number).columns
    print(f"\nNumeric Columns ({len(numeric_cols)}): {list(numeric_cols)}")
    print(f"Categorical Columns ({len(cat_cols)}): {list(cat_cols)}")
    
    return numeric_cols, cat_cols

numeric_cols, cat_cols = enhanced_quality_report(df)

# 3. Save reports to file
with open('data_quality_report.txt', 'w') as f:
    f.write(str(df.describe(include='all')))

# 4. Enhanced Visualization
plt.figure(figsize=(15, 8))
msno.bar(df, color="dodgerblue")
plt.title("Missing Values by Column", pad=20, fontsize=15)
plt.savefig('missing_values.png', bbox_inches='tight')
plt.show()

# 5. Numeric Columns Analysis
if len(numeric_cols) > 0:
    # Correlation heatmap
    plt.figure(figsize=(12, 8))
    sns.heatmap(df[numeric_cols].corr(), annot=True, cmap='coolwarm', center=0)
    plt.title("Correlation Matrix of Numeric Features")
    plt.savefig('correlation_matrix.png', bbox_inches='tight')
    plt.show()
    
    # Pairplot for small number of numeric columns
    if len(numeric_cols) <= 5:
        sns.pairplot(df[numeric_cols])
        plt.savefig('pairplot.png', bbox_inches='tight')
        plt.show()

# 6. Categorical Columns Analysis
for col in cat_cols:
    unique_count = df[col].nunique()
    print(f"\nColumn '{col}': {unique_count} unique values")
    
    if unique_count <= 15:
        plt.figure(figsize=(10, 5))
        if unique_count > 5:
            plt.xticks(rotation=45, ha='right')
        sns.countplot(x=df[col], order=df[col].value_counts().index)
        plt.title(f"Distribution of '{col}'")
        plt.tight_layout()
        plt.savefig(f'categorical_{col}.png', bbox_inches='tight')
        plt.show()

print("\nAnalysis complete! Check generated files:")
print("- data_quality_report.txt")
print("- missing_values.png")
print("- correlation_matrix.png (if numeric columns exist)")
print("- Various categorical distribution plots")