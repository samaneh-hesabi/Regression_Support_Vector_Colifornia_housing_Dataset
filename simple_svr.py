"""
California Housing Price Prediction using Support Vector Regression
A simplified implementation for learning purposes
"""

import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import joblib

def load_and_prepare_data():
    """Load and prepare the California housing dataset"""
    print("Loading California housing dataset...")
    housing = fetch_california_housing()
    X = housing.data
    y = housing.target
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Save the scaler for future use
    joblib.dump(scaler, 'scaler.joblib')
    print("Scaler saved as 'scaler.joblib'")
    
    return X_train_scaled, X_test_scaled, y_train, y_test

def train_model(X_train, y_train):
    """Train the SVR model"""
    print("Training SVR model...")
    model = SVR(kernel='rbf', C=1.0, gamma='scale', epsilon=0.1)
    model.fit(X_train, y_train)
    
    # Save the trained model
    joblib.dump(model, 'trained_model.joblib')
    print("Model saved as 'trained_model.joblib'")
    
    return model

def evaluate_model(model, X_test, y_test):
    """Evaluate the model and print metrics"""
    y_pred = model.predict(X_test)
    
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    
    print("\nModel Performance:")
    print(f"Mean Squared Error: {mse:.4f}")
    print(f"Root Mean Squared Error: {rmse:.4f}")
    print(f"R-squared Score: {r2:.4f}")
    
    return y_pred

def create_visualization(y_test, y_pred):
    """Create and save visualization plots"""
    plt.figure(figsize=(12, 5))
    
    # Actual vs Predicted plot
    plt.subplot(1, 2, 1)
    plt.scatter(y_test, y_pred, alpha=0.5)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.title('Actual vs Predicted Values')
    
    # Residual plot
    plt.subplot(1, 2, 2)
    residuals = y_test - y_pred
    plt.scatter(y_pred, residuals, alpha=0.5)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.xlabel('Predicted Values')
    plt.ylabel('Residuals')
    plt.title('Residual Plot')
    
    plt.tight_layout()
    plt.savefig('results.png')
    plt.close()
    print("\nResults visualization saved as 'results.png'")

def main():
    # Load and prepare data
    X_train, X_test, y_train, y_test = load_and_prepare_data()
    
    # Train model
    model = train_model(X_train, y_train)
    
    # Evaluate model
    y_pred = evaluate_model(model, X_test, y_test)
    
    # Create visualization
    create_visualization(y_test, y_pred)

def prepare_sample_data():
    # Your custom data
    sample_data = {
        'MedInc': [your_values],
        'HouseAge': [your_values],
        'AveRooms': [your_values],
        'AveBedrms': [your_values],
        'Population': [your_values],
        'AveOccup': [your_values],
        'Latitude': [your_values],
        'Longitude': [your_values]
    }
    return pd.DataFrame(sample_data)

if __name__ == "__main__":
    main()