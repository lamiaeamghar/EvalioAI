import pandas as pd
import numpy as np
from unidecode import unidecode
import re
from geopy.geocoders import Nominatim
from sklearn.cluster import DBSCAN
import os

def clean_surface(surface):
    """Clean surface values."""
    if pd.isna(surface):
        return None
    try:
        # Remove non-numeric characters and convert
        return float(re.sub(r'[^\d.,]', '', str(surface)).replace(',', '.')) or None
    except:
        return None

def extract_city(location):
    """Extract city name from location string."""
    if pd.isna(location):
        return None
    
    major_cities = [
        'casablanca', 'rabat', 'marrakech', 'fes', 'tanger', 'agadir',
        'meknes', 'oujda', 'kenitra', 'tetouan', 'safi', 'mohammedia'
    ]
    
    try:
        location_clean = unidecode(str(location)).lower()
        for city in major_cities:
            if city in location_clean:
                return city.title()
        return None
    except:
        return None

def remove_outliers(df, column):
    """Remove outliers using IQR method."""
    if column not in df.columns:
        return df
        
    Q1 = df[column].quantile(0.10)
    Q3 = df[column].quantile(0.90)
    IQR = Q3 - Q1
    return df[~((df[column] < (Q1 - 1.5 * IQR)) | (df[column] > (Q3 + 1.5 * IQR)))]

class DataCleaner:
    def __init__(self):
        self.geolocator = Nominatim(user_agent="morocco_property_cleaner", timeout=10)
        
    def clean_price(self, price):
        """Improved price cleaning with logging"""
        if pd.isna(price):
            return None
        try:
            price_str = str(price)
            # Handle French/English/Arabic formats
            price_str = (price_str.replace('\xa0', '')
                         .replace(' ', '')
                         .replace('DH', '')
                         .replace('د.م.', '')
                         .replace(',', ''))
            return int(re.sub(r'\D', '', price_str)) or None
        except Exception as e:
            print(f"Price cleaning failed for {price}: {str(e)}")
            return None

    def extract_features(self, description):
        """Extract structured features from description"""
        features = {
            'parking': False,
            'pool': False,
            'furnished': False,
            'bedrooms': None,
            'bathrooms': None
        }
        
        if pd.isna(description):
            return features
            
        desc = str(description).lower()
        
        # Extract amenities
        features.update({
            'parking': any(x in desc for x in ['parking', 'garage']),
            'pool': 'piscine' in desc,
            'furnished': 'meublé' in desc or 'furnished' in desc
        })
        
        # Extract bedroom count
        bed_match = re.search(r'(\d+)\s*(chambre|bedroom)', desc)
        if bed_match:
            features['bedrooms'] = int(bed_match.group(1))
            
        # Extract bathroom count
        bath_match = re.search(r'(\d+)\s*(salle|bathroom)', desc)
        if bath_match:
            features['bathrooms'] = int(bath_match.group(1))
            
        return features

    def clean_data(self, input_file, output_file):
        """Enhanced cleaning pipeline"""
        print("Starting data cleaning process...")
        
        try:
            # Load data
            print(f"Loading data from {input_file}...")
            df = pd.read_csv(input_file, encoding='utf-8')
            
            # 1. Clean core fields
            print("Cleaning prices...")
            df['prix'] = df['prix'].apply(self.clean_price).astype('Int64')
            
            print("Cleaning surfaces...")
            df['surface'] = df['surface'].apply(clean_surface)
            
            # Calculate price/m²
            df['prix_m2'] = df['prix'] / df['surface']
            
            # 2. Extract features from description
            print("Extracting features from descriptions...")
            features = df['description'].apply(self.extract_features)
            df = pd.concat([df, features.apply(pd.Series)], axis=1)
            
            # 3. Process locations
            print("Processing locations...")
            df['city'] = df['localisation'].apply(extract_city)
            
            # 4. Handle duplicates
            print("Removing duplicates...")
            df = df.drop_duplicates(
                subset=['prix', 'surface', 'localisation', 'description'],
                keep='first'
            )
            
            # 5. Remove outliers
            print("Removing outliers...")
            df = remove_outliers(df, 'prix_m2')
            df = remove_outliers(df, 'surface')
            
            # Save results
            print(f"Saving cleaned data to {output_file}...")
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            df.to_csv(output_file, index=False)
            
            print("\nCleaning completed successfully!")
            print(f"Original rows: {len(df)}")
            print(f"Cleaned rows: {len(df)}")
            return True
            
        except Exception as e:
            print(f"\nERROR during cleaning: {str(e)}")
            return False

if __name__ == '__main__':
    # Configure paths
    INPUT_PATH = os.path.join('..', '..', 'data', 'raw_data.csv')
    OUTPUT_PATH = os.path.join('..', '..', 'data', 'cleaned_data.csv')
    
    # Run cleaner
    cleaner = DataCleaner()
    success = cleaner.clean_data(INPUT_PATH, OUTPUT_PATH)
    
    if not success:
        print("\nCleaning failed. Please check the error messages.")
    else:
        print("\nData cleaning completed successfully!")