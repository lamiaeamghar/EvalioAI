from flask import Flask, render_template, request, jsonify
import sys
from pathlib import Path

# Add src directory to Python path
project_root = Path(__file__).parent
sys.path.append(str(project_root))
sys.path.append(str(project_root / "src"))
sys.path.append(str(project_root / "src/model"))

from src.model.model import load_data, preprocess_data, train_and_evaluate
from src.model.chat_interface import RealEstateChatbot
from src.model.multilingual_support import assistant
import joblib
import os

app = Flask(__name__)

# Paths configuration
MODEL_PATH = "models/real_estate_model.joblib"
DATA_PATH = "data/data_cleaned_no_outliers.csv"

# Initialize chatbot
bot = None

def initialize_model():
    global bot
    if not os.path.exists(MODEL_PATH):
        print("Training model first...")
        df = load_data(DATA_PATH)
        X_train, X_test, y_train, y_test, preprocessor = preprocess_data(df, 'prix')
        model = train_and_evaluate(X_train, X_test, y_train, y_test)
        joblib.dump(model, MODEL_PATH)
    
    bot = RealEstateChatbot(MODEL_PATH)

initialize_model()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    text = data['text']
    
    price = bot.predict_from_text(text)
    
    if price is None:
        return jsonify({'error': 'Need surface area and room count'})
    
    response = assistant.generate_response(text, {
        'predicted_price': price,
        'features': ['surface', 'pièces', 'ville']
    })
    
    return jsonify({
        'price': price,
        'response': response
    })

if __name__ == '__main__':
    app.run(debug=True)