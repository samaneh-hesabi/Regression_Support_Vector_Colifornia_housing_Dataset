<div style="font-size:2.5em; font-weight:bold; text-align:center; margin-top:20px;">California Housing Price Prediction</div>

# 1. Project Overview
This project implements a Support Vector Regression (SVR) model to predict California housing prices. It provides a clear, beginner-friendly demonstration of machine learning concepts while maintaining professional standards.

# 2. Dataset Description
The California Housing dataset is included in scikit-learn and contains information about housing in California from the 1990 census. 

## 2.1 Features
The dataset includes the following features:
- MedInc: Median income in block group
- HouseAge: Median house age in block group
- AveRooms: Average number of rooms per household
- AveBedrms: Average number of bedrooms per household
- Population: Block group population
- AveOccup: Average number of household members
- Latitude: Block group latitude
- Longitude: Block group longitude

## 2.2 Target Variable
- Target: Median house value in 100,000s of dollars

# 3. Project Structure
```
.
├── simple_svr.py      # Main script for training the SVR model
├── test_model.py      # Script for testing the trained model on new data
├── requirements.txt   # Project dependencies
├── results.png       # Model performance visualizations
├── trained_model.joblib  # Saved trained model
├── scaler.joblib     # Saved feature scaler
├── LICENSE          # MIT License
├── README.md        # Project documentation
├── .git/           # Version control
└── .gitignore      # Git ignore rules
```

# 4. Technical Details

## 4.1 Model Architecture
- Algorithm: Support Vector Regression (SVR)
- Kernel: Radial Basis Function (RBF)
- Parameters:
  - C: 1.0 (Regularization parameter)
  - gamma: 'scale' (Kernel coefficient)
  - epsilon: 0.1 (Margin of tolerance)

## 4.2 Data Preprocessing
- Train-Test Split: 80-20 ratio
- Feature Scaling: StandardScaler
  - Zero mean
  - Unit variance

## 4.3 Model Evaluation
The model's performance is evaluated using:
- Mean Squared Error (MSE)
- Root Mean Squared Error (RMSE)
- R-squared Score (R²)

# 5. Setup and Usage

## 5.1 Prerequisites
- Python 3.8 or higher
- pip (Python package installer)

## 5.2 Installation
1. Clone the repository:
```bash
git clone [repository-url]
cd [repository-name]
```

2. Install required packages:
```bash
pip install -r requirements.txt
```

## 5.3 Running the Model

### Training the Model
Execute the main script to train the model:
```bash
python simple_svr.py
```
This will:
- Load and preprocess the California housing dataset
- Train the SVR model
- Save the trained model and scaler
- Generate performance metrics
- Create visualization plots

### Testing the Model
To test the model on new data:
```bash
python test_model.py
```
This will:
- Load the trained model and scaler
- Make predictions on sample data
- Display the results

# 6. Results
The script generates:
1. Performance Metrics:
   - MSE: Measures average squared difference between predicted and actual values
   - RMSE: Square root of MSE, shows error in the same unit as target variable
   - R²: Indicates proportion of variance in target that's predictable from features

2. Visualizations (results.png):
   - Actual vs Predicted Values Plot: Shows model's prediction accuracy
   - Residual Plot: Displays error distribution

# 7. Model Persistence
The project includes model persistence features:
- `trained_model.joblib`: Contains the trained SVR model
- `scaler.joblib`: Contains the feature scaler used during training

These files allow you to:
- Save trained models for future use
- Load models for predictions without retraining
- Ensure consistent feature scaling across different runs

# 8. Contributing
Contributions are welcome! Please feel free to submit a Pull Request.

# 9. License
This project is open source and available under the MIT License.

# 10. Contact
For any questions or feedback, please open an issue in the GitHub repository.
