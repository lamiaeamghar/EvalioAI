import pandas as pd
import numpy as np
import re
from unidecode import unidecode
import os
from word2number import w2n
from num2words import num2words

class FeatureExtractor:
    def __init__(self):
        # Create French number word mapping for 1-20
        self.fr_numbers = {num2words(n, lang='fr'): n for n in range(1, 21)}
        self.fr_ordinals = {
            'premier': 1, 'première': 1,
            'deuxième': 2, 'second': 2, 'seconde': 2,
            'troisième': 3,
            'quatrième': 4,
            'cinquième': 5,
            'sixième': 6,
            'septième': 7,
            'huitième': 8,
            'neuvième': 9,
            'dixième': 10
        }

    def convert_french_numbers(self, text):
        """Convert French number words to digits in text."""
        def replace_match(match):
            word = match.group(0).lower()
            if word in self.fr_numbers:
                return str(self.fr_numbers[word])
            if word in self.fr_ordinals:
                return str(self.fr_ordinals[word])
            return word
        
        # Match French number words
        pattern = r'\b(' + '|'.join(list(self.fr_numbers.keys()) + list(self.fr_ordinals.keys())) + r')\b'
        return re.sub(pattern, replace_match, text, flags=re.IGNORECASE)

    def normalize_text(self, text):
        """Normalize text by removing accents and converting to lowercase."""
        if pd.isna(text):
            return ""
        text = unidecode(str(text).lower())
        text = ' '.join(text.split())
        return text

    def extract_numeric_value(self, text, patterns):
        """Extract numeric value using multiple patterns."""
        for pattern in patterns:
            match = re.search(pattern, text)
            if match:
                try:
                    # First try direct number conversion
                    return float(match.group(1).replace(',', '.'))
                except:
                    try:
                        # Try word to number conversion
                        return w2n.word_to_num(match.group(1))
                    except:
                        continue
        return None

    def extract_features(self, description):
        """Extract features from property description."""
        features = {
            'surface': None,
            'pièces': None,
            'chambres': None,
            'salles_de_bains': None,
            'standing': None,
            'étage': None,
            'état': None
        }
        
        # Normalize and convert numbers in description
        desc_norm = self.normalize_text(description)
        desc_norm = self.convert_french_numbers(desc_norm)
        
        # Surface patterns (m²)
        surface_patterns = [
            r'(\d+[\.,]?\d*)\s*(?:m2|m\²|mètres?\s*carrés?)',
            r'surface\s*(?:de|d[eu]\s*)?(\d+[\.,]?\d*)',
            r'superficie\s*(?:de\s*)?(\d+[\.,]?\d*)'
        ]
        features['surface'] = self.extract_numeric_value(desc_norm, surface_patterns)
        
        # Room patterns
        room_patterns = [
            r'(\d+)\s*(?:pièces?|p(?:\.|\s*)|pieces?)',
            r't(\d+)',
            r'f(\d+)'
        ]
        features['pièces'] = self.extract_numeric_value(desc_norm, room_patterns)
        
        # Bedroom patterns
        bedroom_patterns = [
            r'(\d+)\s*(?:chambres?|ch(?:\.|\s*)|bedrooms?)',
            r'chambres?\s*(?:de\s*)?(\d+)'
        ]
        features['chambres'] = self.extract_numeric_value(desc_norm, bedroom_patterns)
        
        # Bathroom patterns
        bathroom_patterns = [
            r'(\d+)\s*(?:salles?\s*(?:de\s*)?bains?|sdb|salles?\s*d[\']eau)',
            r'salle\s*de\s*bain\s*(\d+)'
        ]
        features['salles_de_bains'] = self.extract_numeric_value(desc_norm, bathroom_patterns)
        
        # Standing patterns
        standing_map = {
            r'haut\s*standing|luxe|prestige|premium': 'haut standing',
            r'moyen\s*standing|standard': 'moyen standing',
            r'économique|basic|simple': 'économique'
        }
        for pattern, standing in standing_map.items():
            if re.search(pattern, desc_norm):
                features['standing'] = standing
                break
        
        # Floor patterns
        floor_patterns = [
            r'(\d+)(?:e|er|ème|eme)\s*étage',
            r'étage\s*(\d+)',
            r'au\s*(\d+)'
        ]
        features['étage'] = self.extract_numeric_value(desc_norm, floor_patterns)
        
        # State patterns
        state_map = {
            r'neuf|nouveau|récent': 'neuf',
            r'bon\s*état|excellent\s*état': 'bon état',
            r'rénové|refait\s*à\s*neuf': 'rénové',
            r'à\s*rénover|à\s*rafraîchir': 'à rénover'
        }
        for pattern, state in state_map.items():
            if re.search(pattern, desc_norm):
                features['état'] = state
                break
        
        return features

    def process_file(self, input_file, output_file):
        """Process the CSV file to extract features from descriptions."""
        print(f"Loading data from {input_file}...")
        try:
            df = pd.read_csv(input_file)
        except Exception as e:
            print(f"Error loading file: {e}")
            return False
        
        feature_columns = ['surface', 'pièces', 'chambres', 'salles_de_bains', 
                          'standing', 'étage', 'état']
        
        # Initialize missing columns
        for col in feature_columns:
            if col not in df.columns:
                df[col] = np.nan
        
        if 'description' not in df.columns:
            print("Error: 'description' column not found")
            return False
        
        # Find rows needing processing
        needs_processing = df[feature_columns].isnull().any(axis=1) & df['description'].notna()
        print(f"Found {needs_processing.sum()} properties needing feature extraction")
        
        if needs_processing.sum() == 0:
            print("No features to extract - all properties already have features")
            return True
        
        # Process descriptions
        print("Extracting features from descriptions...")
        df.loc[needs_processing, feature_columns] = (
            df.loc[needs_processing, 'description']
            .apply(self.extract_features)
            .apply(pd.Series)
        )
        
        # Save results
        print(f"Saving results to {output_file}...")
        try:
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            df.to_csv(output_file, index=False, encoding='utf-8')
            print("Successfully saved extracted features!")
            print("\nSample results:")
            print(df[feature_columns + ['description']].head(3).to_string())
            return True
        except Exception as e:
            print(f"Error saving file: {e}")
            return False

if __name__ == '__main__':
    current_dir = os.path.dirname(os.path.abspath(__file__))
    INPUT_PATH = os.path.join(current_dir, '..', '..', 'data', 'cleaned_data_with_descriptions.csv')
    OUTPUT_PATH = os.path.join(current_dir, '..', '..', 'data', 'data_with_features.csv')
    
    print(f"Input file: {INPUT_PATH}")
    print(f"Output file: {OUTPUT_PATH}")
    
    extractor = FeatureExtractor()
    success = extractor.process_file(INPUT_PATH, OUTPUT_PATH)
    
    if not success:
        print("\nFeature extraction failed. Please check the error messages.")