import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import os
from typing import Tuple
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor
from optimization import cost_function, optimize_model, plot_error_distribution
from multilingual_support import assistant
from chat_interface import RealEstateChatbot

def load_data(file_path):
    try:
        df = pd.read_csv(file_path)
        print(f"Data loaded successfully. Shape: {df.shape}")
        return df
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

def preprocess_data(df, target_column):
    df['prix_per_surface'] = df['prix'] / df['surface']
    X = df.drop(columns=[target_column])
    y = df[target_column]

    numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns
    categorical_cols = X.select_dtypes(include=['object', 'category']).columns

    if 'chambres' in numerical_cols and 'bedrooms' in numerical_cols:
        X = X.drop(columns=['chambres', 'salles_de_bains'])
        numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns

    numerical_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(handle_unknown='ignore')

    preprocessor = ColumnTransformer([
        ('num', numerical_transformer, numerical_cols),
        ('cat', categorical_transformer, categorical_cols)
    ])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)

    return X_train, X_test, y_train, y_test, preprocessor

def build_model(preprocessor):
    model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', XGBRegressor(
            n_estimators=500,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1
        ))
    ])
    return model

def train_and_evaluate(model, X_train, X_test, y_train, y_test):
    print("\nTraining the model...")
    model.fit(X_train, y_train)

    train_preds = model.predict(X_train)
    test_preds = model.predict(X_test)

    train_rmse = np.sqrt(mean_squared_error(y_train, train_preds))
    test_rmse = np.sqrt(mean_squared_error(y_test, test_preds))
    train_r2 = r2_score(y_train, train_preds)
    test_r2 = r2_score(y_test, test_preds)

    print("\nModel Performance:")
    print(f"Training RMSE: {train_rmse:.2f}")
    print(f"Testing RMSE: {test_rmse:.2f}")
    print(f"Training R²: {train_r2:.4f}")
    print(f"Testing R²: {test_r2:.4f}")

    residuals = test_preds - y_test
    large_errors = residuals[np.abs(residuals) > 200000]
    if len(large_errors) > 0:
        print(f"\nLarge prediction errors found: {len(large_errors)} samples")

    return model, test_preds, residuals

def save_model(model, file_path):
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        joblib.dump(model, file_path)
        print(f"\nModel saved to {file_path}")
    except Exception as e:
        print(f"Error saving model: {e}")

def chat_interface(model_path):
    bot = RealEstateChatbot(model_path)
    print("🏠 Real Estate Chatbot (type 'quit' to exit)")
    print("Example: 'Appartement de 120m² avec 3 pièces à Casablanca'\n")

    while True:
        user_input = input("You: ")
        if user_input.lower() in ['quit', 'exit']:
            break
        response = bot.respond(user_input)
        print("Bot:", response)


def main():
    if 'COLAB_GPU' in os.environ:
        base_dir = "/content/drive/MyDrive/Advanced EvalioIA"
    elif os.path.exists("/kaggle/working"):
        base_dir = "/kaggle/input/evalioai/pytorch/default/2/Advanced EvalioIA"
    else:
        base_dir = r"C:\\Users\\hp\\Documents\\INOCOD\\Advanced EvalioIA"

    data_path = os.path.join(base_dir, "data", "data_cleaned_no_outliers.csv")
    model_path = os.path.join(base_dir, "models", "real_estate_model.joblib")
    target_column = 'prix'

    df = load_data(data_path)
    if df is None:
        return

    X_train, X_test, y_train, y_test, preprocessor = preprocess_data(df, target_column)

    print("\nData Split Summary:")
    print(f"Training set size: {len(X_train)} samples")
    print(f"Test set size: {len(X_test)} samples")

    model = build_model(preprocessor)

    trained_model, test_predictions, residuals = train_and_evaluate(
        model, X_train, X_test, y_train, y_test)

    train_cost = cost_function(trained_model, X_train, y_train)
    test_cost = cost_function(trained_model, X_test, y_test)
    print(f"\nTraining Cost: {train_cost:.2f}")
    print(f"Testing Cost: {test_cost:.2f}")

    save_model(trained_model, model_path)

    results_df = pd.DataFrame({
        'Actual': y_test,
        'Predicted': test_predictions,
        'Difference': residuals,
        'Percentage_Difference': (residuals / y_test) * 100
    })
    print("\nSample Predictions vs Actual:")
    print(results_df.head(10).to_string())

    results_df.to_csv(os.path.join(os.path.dirname(model_path), 'prediction_results.csv'), index=False)
    print(f"\nPrediction results saved to {os.path.join(os.path.dirname(model_path), 'prediction_results.csv')}")

    plot_error_distribution(y_test, test_predictions)

    if hasattr(trained_model.named_steps['regressor'], 'feature_importances_'):
        try:
            feature_names = []
            feature_names.extend(X_train.select_dtypes(include=['int64', 'float64']).columns)
            ohe = trained_model.named_steps['preprocessor'].named_transformers_['cat']
            cat_features = ohe.get_feature_names_out(X_train.select_dtypes(include=['object', 'category']).columns)
            feature_names.extend(cat_features)

            importances = trained_model.named_steps['regressor'].feature_importances_
            indices = np.argsort(importances)[-10:]

            plt.figure(figsize=(10, 6))
            plt.title('Top 10 Feature Importances')
            plt.barh(range(len(indices)), importances[indices], align='center')
            plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
            plt.xlabel('Relative Importance')
            plt.tight_layout()
            plt.show()
        except Exception as e:
            print(f"\nCould not plot feature importances: {e}")

    if input("\nStart chat interface? (y/n): ").lower() == 'y':
        chat_interface(model_path)

if __name__ == "__main__":
    main()
