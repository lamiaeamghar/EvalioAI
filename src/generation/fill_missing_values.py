import pandas as pd
import numpy as np

def fill_missing_values(df):
    """
    Fill missing values in the real estate dataset using realistic imputation methods.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame with missing values
    
    Returns:
    pd.DataFrame: DataFrame with missing values filled
    """
    # Make a copy to avoid modifying the original
    df_filled = df.copy()
    
    # 1. Fill city from localisation where possible
    df_filled['city'] = df_filled['city'].fillna(
        df_filled['localisation'].str.split(',').str[-1].str.strip()
    )
    
    # 2. Fill related columns that mean the same thing
    df_filled['bedrooms'] = df_filled['bedrooms'].fillna(df_filled['chambres'])
    df_filled['bathrooms'] = df_filled['bathrooms'].fillna(df_filled['salles_de_bains'])
    
    # 3. For numerical columns, fill with median grouped by city and type
    num_cols = ['surface', 'pièces', 'chambres', 'salles_de_bains', 'bedrooms', 'bathrooms']
    for col in num_cols:
        df_filled[col] = df_filled[col].fillna(
            df_filled.groupby(['city', 'type'])[col].transform('median'))
        # If still missing, fill with overall median
        df_filled[col] = df_filled[col].fillna(df_filled[col].median())
    
    # 4. For categorical columns, fill with mode grouped by city and type
    cat_cols = ['standing', 'étage', 'état']
    for col in cat_cols:
        df_filled[col] = df_filled[col].fillna(
            df_filled.groupby(['city', 'type'])[col].transform(
                lambda x: x.mode()[0] if not x.mode().empty else np.nan))
        # If still missing, fill with overall mode
        mode_val = df_filled[col].mode()[0] if not df_filled[col].mode().empty else np.nan
        df_filled[col] = df_filled[col].fillna(mode_val)
    
    # 5. Fill caractéristiques with empty string if missing
    df_filled['caractéristiques'] = df_filled['caractéristiques'].fillna('')
    
    # 6. Calculate pièces from surface if missing (average surface per room)
    mask = df_filled['pièces'].isna() & df_filled['surface'].notna()
    if not df_filled['pièces'].isna().all():  # Only calculate if some pièces values exist
        avg_surface_per_room = df_filled['surface'].mean() / df_filled['pièces'].mean()
        df_filled.loc[mask, 'pièces'] = np.round(df_filled.loc[mask, 'surface'] / avg_surface_per_room)
    
    # 7. Ensure logical relationships between columns
    df_filled['bedrooms'] = df_filled[['bedrooms', 'pièces']].min(axis=1)
    df_filled['chambres'] = df_filled[['chambres', 'pièces']].min(axis=1)
    df_filled['bathrooms'] = df_filled[['bathrooms', 'bedrooms']].min(axis=1)
    df_filled['salles_de_bains'] = df_filled[['salles_de_bains', 'chambres']].min(axis=1)
    
    # 8. Fill étage with reasonable defaults
    mask = df_filled['étage'].isna()
    df_filled.loc[mask & (df_filled['type'] == 'Villa'), 'étage'] = 'RDC'
    df_filled.loc[mask & (df_filled['type'] == 'Appartement'), 'étage'] = '1'
    df_filled['étage'] = df_filled['étage'].fillna('1')  # Default for other types
    
    # 9. Fill état based on standing
    mask = df_filled['état'].isna()
    df_filled.loc[mask & (df_filled['standing'] == 'haut standing'), 'état'] = 'Neuf'
    df_filled.loc[mask, 'état'] = df_filled['état'].fillna('Bon état')
    
    return df_filled

def main():
    # Load the data
    input_path = '../../data/data_with_features.csv'
    try:
        df = pd.read_csv(input_path)
        print("Data loaded successfully. Shape:", df.shape)
    except Exception as e:
        print(f"Error loading data: {e}")
        return
    
    # Fill missing values
    df_filled = fill_missing_values(df)
    
    # Verify no missing values remain
    print("\nMissing values after imputation:")
    print(df_filled.isnull().sum())
    
    # Save to new file
    output_path = '../../data/data_with_features_filled.csv'
    try:
        df_filled.to_csv(output_path, index=False, encoding='utf-8-sig')
        print(f"\nData saved successfully to: {output_path}")
        print("Final shape:", df_filled.shape)
    except Exception as e:
        print(f"Error saving data: {e}")

if __name__ == "__main__":
    main();