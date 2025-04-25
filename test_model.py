"""
Test the trained SVR model on new data
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib
import matplotlib.pyplot as plt

def load_model_and_scaler():
    """Load the trained model and scaler"""
    try:
        model = joblib.load('trained_model.joblib')
        scaler = joblib.load('scaler.joblib')
        print("Model and scaler loaded successfully!")
        return model, scaler
    except FileNotFoundError:
        print("Error: Model or scaler file not found. Please run simple_svr.py first to train the model.")
        return None, None

def prepare_sample_data():
    """
    Prepare sample data in the same format as the California Housing dataset
    You can replace this with your own data loading logic
    """
    # Sample data format (replace with your actual data)
    sample_data = {
        'MedInc': [8.3252, 7.2574, 5.6431],
        'HouseAge': [41.0, 21.0, 52.0],
        'AveRooms': [6.984127, 6.238137, 8.288136],
        'AveBedrms': [1.023810, 0.971880, 1.073446],
        'Population': [322.0, 2401.0, 1157.0],
        'AveOccup': [2.555556, 2.109842, 3.741935],
        'Latitude': [37.88, 37.86, 37.85],
        'Longitude': [-122.23, -122.22, -122.24]
    }
    
    return pd.DataFrame(sample_data)

def predict_house_prices(model, scaler, new_data):
    """Make predictions on new data"""
    if model is None or scaler is None:
        return None
    
    # Scale the features using the same scaler used during training
    new_data_scaled = scaler.transform(new_data)
    
    # Make predictions
    predictions = model.predict(new_data_scaled)
    
    return predictions

def display_results(new_data, predictions):
    """Display the input data and corresponding predictions"""
    if predictions is None:
        return
    
    results = new_data.copy()
    results['Predicted_Price'] = predictions * 100000  # Convert back to dollars
    
    print("\nPredictions for new data:")
    print("=" * 80)
    for idx, row in results.iterrows():
        print(f"\nHouse {idx + 1}:")
        print(f"Input Features:")
        for col in row.index[:-1]:  # Exclude the prediction column
            print(f"  - {col}: {row[col]}")
        print(f"Predicted Price: ${row['Predicted_Price']:,.2f}")
    print("=" * 80)

def main():
    # Load the trained model and scaler
    model, scaler = load_model_and_scaler()
    if model is None or scaler is None:
        return
    
    # Prepare new data (replace this with your own data)
    new_data = prepare_sample_data()
    
    # Make predictions
    predictions = predict_house_prices(model, scaler, new_data)
    
    # Display results
    display_results(new_data, predictions)

if __name__ == "__main__":
    main() 