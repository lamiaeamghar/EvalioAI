import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

def load_data(file_path):
    """Load the dataset from the specified file path."""
    try:
        df = pd.read_csv(file_path)
        print(f"Data loaded successfully. Shape: {df.shape}")
        return df
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

def detect_and_remove_outliers(df, numerical_cols):
    """
    Detect and remove outliers using a combination of methods with moderate thresholds.
    Returns a cleaned DataFrame and removal statistics.
    """
    # Make a copy of the original data
    df_clean = df.copy()
    removal_stats = {}
    
    # 1. Remove using IQR method with moderate threshold (1.2 instead of 1.5)
    for col in numerical_cols:
        Q1 = df_clean[col].quantile(0.25)
        Q3 = df_clean[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.2 * IQR
        upper_bound = Q3 + 1.2 * IQR
        
        # Count outliers before removal
        outlier_count = ((df_clean[col] < lower_bound) | (df_clean[col] > upper_bound)).sum()
        removal_stats[f'{col}_IQR_outliers'] = outlier_count
        
        # Remove outliers
        df_clean = df_clean[df_clean[col].between(lower_bound, upper_bound)]
    
    # 2. Remove using Z-score with moderate threshold (2.8 instead of 3)
    for col in numerical_cols:
        if col in df_clean.columns:  # Column might have been removed in previous step
            z_scores = np.abs(stats.zscore(df_clean[col]))
            outlier_mask = z_scores > 2.8
            
            # Count outliers before removal
            removal_stats[f'{col}_z_outliers'] = outlier_mask.sum()
            
            # Remove outliers
            df_clean = df_clean[~outlier_mask]
    
    # 3. Remove multivariate outliers using Mahalanobis distance with moderate threshold (13 instead of 15)
    if len(numerical_cols) >= 2:
        try:
            # Use only columns that still exist in df_clean
            remaining_cols = [col for col in numerical_cols if col in df_clean.columns]
            if len(remaining_cols) >= 2:
                df_numeric = df_clean[remaining_cols].dropna()
                cov_matrix = df_numeric.cov().values
                inv_cov_matrix = np.linalg.inv(cov_matrix)
                mean_vector = df_numeric.mean().values
                
                mahalanobis_dist = []
                for _, row in df_numeric.iterrows():
                    diff = row.values - mean_vector
                    dist = np.sqrt(np.dot(np.dot(diff, inv_cov_matrix), diff.T))
                    mahalanobis_dist.append(dist)
                
                mahalanobis_dist = np.array(mahalanobis_dist)
                outlier_mask = mahalanobis_dist > 13
                
                # Count outliers before removal
                removal_stats['multivariate_outliers'] = outlier_mask.sum()
                
                # Remove outliers
                df_clean = df_clean.loc[df_numeric.index[~outlier_mask]]
        except Exception as e:
            print(f"Error in Mahalanobis calculation: {e}")
    
    return df_clean, removal_stats

def visualize_cleaned_data(original_df, cleaned_df, numerical_cols):
    """Compare original and cleaned data distributions."""
    for col in numerical_cols:
        plt.figure(figsize=(14, 6))
        
        plt.subplot(1, 2, 1)
        sns.boxplot(x=original_df[col])
        plt.title(f'Original {col}')
        
        plt.subplot(1, 2, 2)
        sns.boxplot(x=cleaned_df[col])
        plt.title(f'Cleaned {col}')
        
        plt.tight_layout()
        plt.show()

def save_cleaned_data(df, file_path):
    """Save the cleaned data to a CSV file."""
    try:
        df.to_csv(file_path, index=False)
        print(f"Cleaned data saved to: {file_path}")
    except Exception as e:
        print(f"Error saving cleaned data: {e}")

def main():
    # Configuration
    input_file = r'C:\Users\hp\Documents\INOCOD\Advanced EvalioIA\data\data_cleaned_no_outliers.csv'
    output_file = r'C:\Users\hp\Documents\INOCOD\Advanced EvalioIA\data\data_cleaned_no_outliers.csv'
    
    # Define numerical columns to analyze
    numerical_cols = [
        'surface', 'pièces', 'chambres', 'salles_de_bains', 
        'bedrooms', 'bathrooms', 'prix', 'prix_m2'
    ]
    
    # Load data
    df = load_data(input_file)
    if df is None:
        return
    
    # Clean data by removing outliers
    print("\nCleaning data by removing outliers...")
    df_clean, removal_stats = detect_and_remove_outliers(df, numerical_cols)
    
    # Print removal statistics
    print("\nOutliers Removed:")
    for col, count in removal_stats.items():
        print(f"{col}: {count} removed")
    
    # Show before/after counts
    print(f"\nOriginal data count: {len(df)}")
    print(f"Cleaned data count: {len(df_clean)}")
    print(f"Percentage removed: {(1 - len(df_clean)/len(df))*100:.2f}%")
    
    # Visualize the cleaning results
    print("\nGenerating visualizations...")
    visualize_cleaned_data(df, df_clean, numerical_cols)
    
    # Save cleaned data
    save_cleaned_data(df_clean, output_file)
    
    # Summary report
    print("\nData Cleaning Summary:")
    print(f"Total properties analyzed: {len(df)}")
    print(f"Properties after cleaning: {len(df_clean)}")
    print(f"Total removed: {len(df) - len(df_clean)} ({(1 - len(df_clean)/len(df))*100:.2f}%)")

if __name__ == "__main__":
    main()