import re
import pandas as pd
import joblib
from sklearn.compose import ColumnTransformer
import sklearn

class RealEstateChatbot:
    def __init__(self, model_path):
        # Load the model
        self.model = joblib.load(model_path)
        self.data = {}
        self.required_features = ['type', 'surface', 'pièces', 'ville']
        self.default_values = {
            'type': 'Appartement',
            'titre': '',
            'localisation': '',
            'chambres': 2.0,
            'salles_de_bains': 1.0,
            'caractéristiques': '',
            'description': '',
            'prix_m2': 10000.0,  # Default price per square meter
            'prix_per_surface': 10000.0,  # Added to match expected column
            'parking': False,
            'pool': False,
            'furnished': False,
            'bedrooms': 2.0,
            'bathrooms': 1.0,
            'standing': 'moyen standing',
            'étage': 1.0,
            'état': 'neuf'
        }
        self.types_list = ['Appartement', 'Maison', 'Villa', 'Riad', 'Local commercial', 'Bureau', 'Commerce', 'Ferme', 'Terrain']
        self.standings = ['bas standing', 'moyen standing', 'haut standing']
        self.etats = ['neuf', 'récent', 'ancien', 'à rénover']

        # Inspect the pipeline to confirm expected columns (for debugging)
        if hasattr(self.model, 'named_steps'):
            for step_name, step in self.model.named_steps.items():
                if isinstance(step, ColumnTransformer):
                    print(f"ColumnTransformer columns: {step.transformers}")

    def parse_input(self, text):
        # Extract type
        found_type = None
        for t in self.types_list:
            if t.lower() in text.lower():
                found_type = t
                break
        self.data['type'] = found_type if found_type else self.data.get('type', self.default_values['type'])

        # Extract surface (m²)
        surf = re.search(r'(\d+(?:[.,]\d+)?)\s?m[²2]?', text)
        if surf:
            surf_val = surf.group(1).replace(',', '.')
            self.data['surface'] = float(surf_val)
        else:
            self.data['surface'] = self.data.get('surface')

        # Extract pièces or chambres
        pieces = re.search(r'(\d+)\s?pièces?', text)
        if not pieces:
            pieces = re.search(r'(\d+)\s?chambres?', text)
        if pieces:
            self.data['pièces'] = int(pieces.group(1))
        else:
            self.data['pièces'] = self.data.get('pièces')

        # Extract ville after "à" or "dans"
        ville_match = re.search(r'(?:à|dans)\s([A-Za-zéèêàù\s\-]+)', text)
        if ville_match:
            ville = ville_match.group(1).strip()
            self.data['ville'] = ville
        else:
            self.data['ville'] = self.data.get('ville')

    def check_missing(self):
        missing = []
        for feature in self.required_features:
            if feature not in self.data or self.data[feature] is None:
                missing.append(feature)
        return missing

    def validate_input(self, feature, value):
        if feature == 'surface':
            try:
                val = float(value.replace(',', '.'))
                if val <= 0:
                    return False, "La surface doit être un nombre positif."
                return True, val
            except:
                return False, "Veuillez entrer un nombre valide pour la surface."
        elif feature == 'pièces':
            try:
                val = int(value)
                if val <= 0:
                    return False, "Le nombre de pièces doit être un entier positif."
                return True, val
            except:
                return False, "Veuillez entrer un entier valide pour le nombre de pièces."
        elif feature == 'ville':
            if len(value.strip()) == 0:
                return False, "La ville ne peut pas être vide."
            return True, value.strip()
        elif feature == 'type':
            val = value.strip().title()
            if val in self.types_list:
                return True, val
            else:
                return False, f"Type invalide. Choisissez parmi: {', '.join(self.types_list)}"
        elif feature == 'standing':
            val = value.strip().lower()
            if val in self.standings:
                return True, val
            else:
                return False, f"Standing invalide. Choisissez parmi: {', '.join(self.standings)}"
        elif feature == 'état':
            val = value.strip().lower()
            if val in self.etats:
                return True, val
            else:
                return False, f"État invalide. Choisissez parmi: {', '.join(self.etats)}"
        else:
            return True, value

    def ask_for_feature(self, feature):
        if feature == 'type':
            return f"Quel est le type du bien ? (Options: {', '.join(self.types_list)})"
        elif feature == 'surface':
            return "Quelle est la surface en m² ?"
        elif feature == 'pièces':
            return "Combien y a-t-il de pièces ?"
        elif feature == 'ville':
            return "Dans quelle ville se situe le bien ?"
        elif feature == 'standing':
            return f"Quel est le standing ? (Options: {', '.join(self.standings)})"
        elif feature == 'état':
            return f"Quel est l'état du bien ? (Options: {', '.join(self.etats)})"
        else:
            return f"Veuillez fournir la valeur pour {feature}."

    def fill_defaults(self):
        for k, v in self.default_values.items():
            if k not in self.data or self.data[k] is None:
                self.data[k] = v

    def model_predict(self):
        # Define expected columns based on the model's pipeline
        expected_cols = ['type', 'titre', 'prix', 'localisation', 'surface', 'pièces', 'chambres',
                         'salles_de_bains', 'caractéristiques', 'description', 'prix_m2', 'prix_per_surface',
                         'parking', 'pool', 'furnished', 'bedrooms', 'bathrooms', 'city', 'standing', 'étage', 'état']

        input_dict = {}
        for col in expected_cols:
            if col == 'city':
                input_dict[col] = self.data.get('ville', 'Marrakech')
            elif col in ['prix_m2', 'prix_per_surface']:
                # Use the same default value for both prix_m2 and prix_per_surface
                input_dict[col] = self.data.get('prix_m2', self.default_values.get('prix_m2', 10000.0))
            else:
                input_dict[col] = self.data.get(col, self.default_values.get(col, 0))

        input_df = pd.DataFrame([input_dict])

        # Verify that all expected columns are present
        missing_cols = set(expected_cols) - set(input_df.columns)
        if missing_cols:
            raise ValueError(f"Missing columns in input DataFrame: {missing_cols}")

        prediction = self.model.predict(input_df)[0]
        return round(prediction, 2)

    def respond(self, user_input):
        # If waiting for a missing feature value, validate and store it
        if hasattr(self, 'waiting_for'):
            feature = self.waiting_for
            valid, val_or_msg = self.validate_input(feature, user_input)
            if valid:
                self.data[feature] = val_or_msg
                del self.waiting_for
            else:
                return val_or_msg  # Error message, ask again

        else:
            # Parse input text
            self.parse_input(user_input)

        # Check missing features
        missing = self.check_missing()
        if missing:
            # Ask for first missing feature
            self.waiting_for = missing[0]
            return self.ask_for_feature(missing[0])

        # All required data collected: fill defaults, predict, and respond
        self.fill_defaults()
        price = self.model_predict()
        self.data = {}  # Reset for next session
        return f"Prix estimé : {price} DH"

def main():
    model_path = "/content/drive/MyDrive/Advanced EvalioIA/models/real_estate_model.joblib"  # Updated path
    bot = RealEstateChatbot(model_path)

    print("🏠 Chatbot Immobilier (tapez 'quit' pour sortir)")
    while True:
        user_input = input("Vous: ")
        if user_input.lower() == 'quit':
            print("Au revoir !")
            break
        response = bot.respond(user_input)
        print("Bot:", response)

if __name__ == "__main__":
    main()