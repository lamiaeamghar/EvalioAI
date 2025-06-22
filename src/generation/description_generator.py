import pandas as pd
import numpy as np
from unidecode import unidecode
import re
import random
import os

class DescriptionGenerator:
    def __init__(self):
        pass
    
    def clean_numeric(self, value):
        """Convert string numbers to proper numeric values"""
        if pd.isna(value):
            return np.nan
        try:
            if isinstance(value, str):
                # Remove all non-numeric characters except decimal point
                cleaned = re.sub(r'[^\d.]', '', value)
                return float(cleaned) if cleaned else np.nan
            return float(value)
        except:
            return np.nan
    
    def generate_description(self, row):
        """Generate a property description from features."""
        description_parts = []
        
        # Property type (default to "Bien immobilier" if type is missing)
        prop_type = row.get('type', 'Bien immobilier')
        if pd.isna(prop_type):
            prop_type = 'Bien immobilier'
        
        # Start with property type and surface
        surface = self.clean_numeric(row.get('surface'))
        if pd.notna(surface):
            description_parts.append(f"{prop_type} de {surface:.0f}m²")
        else:
            description_parts.append(prop_type)
        
        # Add number of rooms
        chambres = self.clean_numeric(row.get('chambres'))
        if pd.notna(chambres):
            description_parts.append(f"{chambres:.0f} chambre{'s' if chambres > 1 else ''}")
        
        # Add number of bathrooms
        sdb = self.clean_numeric(row.get('salles_de_bains'))
        if pd.notna(sdb):
            description_parts.append(f"{sdb:.0f} salle{'s' if sdb > 1 else ''} de bain")
        
        # Add location
        if pd.notna(row.get('localisation')):
            description_parts.append(f"situé à {row['localisation']}")
        
        # Add price if available
        prix = self.clean_numeric(row.get('prix'))
        if pd.notna(prix):
            price_str = f"{prix:,.0f} DH".replace(",", " ")
            description_parts.append(f"au prix de {price_str}")
        
        # Add price per m² if available
        prix_m2 = self.clean_numeric(row.get('prix_m2'))
        if pd.notna(prix_m2):
            price_m2_str = f"{prix_m2:,.0f} DH/m²".replace(",", " ")
            description_parts.append(f"({price_m2_str})")
        
        # Add random positive features
        features = []
        if random.random() > 0.5:
            features.extend([
                "lumineux", "bien agencé", "rénové", "moderne",
                "calme", "ensoleillé", "spacieux", "vue dégagée"
            ])
        
        # Add standing based on price per m²
        if pd.notna(prix_m2):
            if prix_m2 > 20000:
                features.append("haut standing")
            elif prix_m2 > 15000:
                features.append("standing")
            elif prix_m2 > 10000:
                features.append("moyen standing")
        
        # Add random selected features
        if features:
            selected_features = random.sample(features, min(3, len(features)))
            description_parts.append(", ".join(selected_features))
        
        # Combine all parts
        description = " ".join(description_parts)
        
        # Add a call to action
        cta = random.choice([
            "À visiter rapidement !",
            "Belle opportunité !",
            "À ne pas manquer !",
            "Contactez-nous pour plus d'informations.",
            "Disponible immédiatement."
        ])
        description += f" {cta}"
        
        return description

    def generate_descriptions(self, input_file, output_file):
        """Generate descriptions for all properties."""
        print("Starting description generation process...")
        
        try:
            # Load data with explicit type conversion
            print(f"Loading data from {input_file}...")
            df = pd.read_csv(input_file)
            
            # Convert numeric columns
            numeric_cols = ['surface', 'chambres', 'salles_de_bains', 'prix', 'prix_m2']
            for col in numeric_cols:
                if col in df.columns:
                    df[col] = df[col].apply(self.clean_numeric)
            
            # Generate descriptions
            print("\nGenerating descriptions...")
            if 'description' not in df.columns or df['description'].isna().all():
                df['description'] = df.apply(self.generate_description, axis=1)
            else:
                print("Descriptions already exist, merging with existing ones")
                mask = df['description'].isna()
                df.loc[mask, 'description'] = df[mask].apply(self.generate_description, axis=1)
            
            # Save results
            print(f"\nSaving data with descriptions to {output_file}...")
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            df.to_csv(output_file, index=False)
            
            print("\nDescription generation completed successfully!")
            print(f"Processed {len(df)} properties")
            print("\nSample generated descriptions:")
            for desc in df['description'].head(3):
                print(f"\n- {desc}")
                
            return True
            
        except Exception as e:
            print(f"\nERROR during generation: {str(e)}")
            return False

if __name__ == '__main__':
    # Configure paths - using absolute paths for better reliability
    current_dir = os.path.dirname(os.path.abspath(__file__))
    INPUT_PATH = os.path.abspath(os.path.join(current_dir, '..', '..', 'data', 'cleaned_data.csv'))
    OUTPUT_PATH = os.path.abspath(os.path.join(current_dir, '..', '..', 'data', 'cleaned_data_with_descriptions.csv'))
    
    print(f"Input file: {INPUT_PATH}")
    print(f"Output file: {OUTPUT_PATH}")
    
    # Run generator
    generator = DescriptionGenerator()
    success = generator.generate_descriptions(INPUT_PATH, OUTPUT_PATH)
    
    if not success:
        print("\nDescription generation failed. Please check the error messages.")