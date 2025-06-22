#optimization.py
import numpy as np
from sklearn.metrics import mean_squared_error
from skopt import BayesSearchCV
from skopt.space import Integer, Real
import matplotlib.pyplot as plt

def cost_function(model, X, y_true, lambda_reg=0.5):
    """
    Custom cost function combining prediction error and model complexity
    
    Parameters:
    - model: Trained model
    - X: Features
    - y_true: True target values
    - lambda_reg: Regularization parameter
    
    Returns:
    - Total cost (prediction error + regularization term)
    """
    print("\nCalculating cost function...")
    
    # Prediction error (MSE)
    y_pred = model.predict(X)
    prediction_error = mean_squared_error(y_true, y_pred)
    print(f"Prediction error (MSE): {prediction_error:.4f}")
    
    # Model complexity term (using number of leaves in RandomForest)
    if hasattr(model.named_steps['regressor'], 'estimators_'):
        n_leaves = sum(tree.tree_.n_leaves for tree in model.named_steps['regressor'].estimators_)
        complexity_term = n_leaves * lambda_reg
        print(f"Complexity term (n_leaves={n_leaves} * lambda={lambda_reg}): {complexity_term:.2f}")
    else:
        complexity_term = 0
        print("No complexity term calculated (not a RandomForest model)")
    
    total_cost = prediction_error + complexity_term
    print(f"Total cost: {total_cost:.4f}")
    
    return total_cost

def optimize_model(X_train, y_train, preprocessor, n_iter=30):
    """
    Bayesian optimization of hyperparameters
    
    Parameters:
    - X_train: Training features
    - y_train: Training targets
    - preprocessor: Preprocessing pipeline
    - n_iter: Number of optimization iterations
    
    Returns:
    - Optimized model
    """
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.pipeline import Pipeline
    
    print("\nStarting model optimization...")
    print(f"Number of optimization iterations: {n_iter}")
    print(f"Training data shape: {X_train.shape}")
    
    # Define the model pipeline
    model = Pipeline([
        ('preprocessor', preprocessor),
        ('regressor', RandomForestRegressor(random_state=42, n_jobs=-1))
    ])
    
    # Define search space
    search_spaces = {
        'regressor__n_estimators': Integer(50, 500),
        'regressor__max_depth': Integer(3, 25),
        'regressor__min_samples_split': Integer(2, 10),
        'regressor__min_samples_leaf': Integer(1, 5),
        'regressor__max_features': Real(0.1, 0.9, prior='uniform')
    }
    
    # Bayesian optimization
    opt = BayesSearchCV(
        estimator=model,
        search_spaces=search_spaces,
        n_iter=n_iter,
        cv=5,
        scoring='neg_mean_squared_error',
        random_state=42,
        n_jobs=-1
    )
    
    # Run optimization
    print("\nRunning optimization... (this may take some time)")
    opt.fit(X_train, y_train)
    
    print("\nOptimization completed!")
    print("\nBest hyperparameters found:")
    print(opt.best_params_)
    
    return opt.best_estimator_

def plot_error_distribution(y_true, y_pred):
    """Plot the distribution of prediction errors"""
    print("\nPlotting error distribution...")
    errors = y_pred - y_true
    plt.figure(figsize=(10, 6))
    plt.hist(errors, bins=50)
    plt.title('Distribution of Prediction Errors')
    plt.xlabel('Prediction Error (Predicted - Actual)')
    plt.ylabel('Frequency')
    plt.axvline(x=0, color='r', linestyle='--')
    plt.show()
    print("Plot displayed.")