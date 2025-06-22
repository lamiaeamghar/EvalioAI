import sys
sys.path.append("/content/Advanced EvalioIA/src/model")
from flask import Flask, render_template, request, jsonify
from flask_ngrok import run_with_ngrok
from pyngrok import ngrok
import joblib
import os
import sys
from pathlib import Path

if "google.colab" in sys.modules:
    base_dir = "/content/drive/MyDrive/Advanced EvalioIA"
elif os.path.exists("/kaggle/working"):
    base_dir = "/kaggle/input/evalioai/pytorch/default/2/Advanced EvalioIA"
elif os.name == "nt":  # Windows
    base_dir = r"C:\Users\hp\Documents\INOCOD\Advanced EvalioIA"
else:
    # Fallback par défaut = Colab
    base_dir = "/content/drive/MyDrive/Advanced EvalioIA"

# Ajout des chemins au PYTHONPATH
project_root = Path(base_dir)
sys.path.append(str(project_root))
sys.path.append(str(project_root / "src"))
sys.path.append(str(project_root / "src/model"))

# Importations des modules internes
try:
    from src.model.model import load_data, preprocess_data, train_and_evaluate, build_model
    from src.model.chat_interface import RealEstateChatbot
    from src.model.multilingual_support import assistant
except ImportError as e:
    print(f"[ERREUR IMPORT] : {e}")
    raise

# Initialisation Flask
app = Flask(__name__)
run_with_ngrok(app)

# Chemins
DATA_PATH = os.path.join(base_dir, "data", "data_cleaned_no_outliers.csv")
MODEL_PATH = os.path.join(base_dir, "models", "real_estate_model.joblib")

# Chargement du modèle
bot = None

def initialize_model():
    global bot
    try:
        if not os.path.exists(MODEL_PATH):
            print("Entraînement du modèle...")
            df = load_data(DATA_PATH)
            X_train, X_test, y_train, y_test, preprocessor = preprocess_data(df, 'prix')
            model, _, _ = train_and_evaluate(
                build_model(preprocessor), X_train, X_test, y_train, y_test
            )
            joblib.dump(model, MODEL_PATH)
        print("Modèle prêt.")
        bot = RealEstateChatbot(MODEL_PATH)
    except Exception as e:
        print(f"[ERREUR INITIALISATION BOT] : {e}")
        raise

# Routes Flask
@app.route('/')
def home():
    try:
        return render_template('index.html')
    except Exception as e:
        return f"Erreur de rendu : {e}", 500


@app.route("/predict", methods=["POST"])
@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        text = data.get("text", "")
        response_text = bot.respond(text)

        return jsonify({'response': response_text})
    except Exception as e:
        return jsonify({'error': f'Erreur de prédiction : {e}'}), 500


# Initialisation modèle et lancement
try:
    initialize_model()
except Exception as e:
    print(f"[ERREUR INIT GLOBALE] : {e}")
    raise

# Lancement serveur
public_url = ngrok.connect(5000)
print(f"Ngrok public URL (ouvre-le dans ton navigateur) : {public_url}")

app.run()
