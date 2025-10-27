from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import pickle
import os
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error

app = Flask(__name__)

# Global variables for the model
model = None
feature_names = None
scaler = None

def load_and_train_model():
    """Load data and train the model"""
    global model, feature_names
    
    try:
        # Load the real house data
        csv_path = os.path.join(os.path.dirname(__file__), 'real_house_data.csv')
        print(f"Loading data from: {csv_path}")
        df = pd.read_csv(csv_path)
        print(f"Data loaded successfully. Shape: {df.shape}")
    except FileNotFoundError:
        print("Error: real_house_data.csv not found!")
        print("Make sure the CSV file is in the same directory as app.py")
        return None, None
    except Exception as e:
        print(f"Error loading data: {e}")
        return None, None
    
    # Drop address column
    df = df.drop('Address', axis=1)
    
    # Remove outliers using IQR method
    Q1 = df['Price'].quantile(0.25)
    Q3 = df['Price'].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    df_clean = df[(df['Price'] >= lower_bound) & (df['Price'] <= upper_bound)]
    
    # Prepare features and target
    X = df_clean.drop('Price', axis=1)
    y = df_clean['Price']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train model
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # Store feature names
    feature_names = list(X.columns)
    
    # Calculate model performance
    train_score = model.score(X_train, y_train)
    test_score = model.score(X_test, y_test)
    
    print(f"Model trained successfully!")
    print(f"Training R² Score: {train_score:.4f}")
    print(f"Testing R² Score: {test_score:.4f}")
    
    return model, feature_names

@app.route('/')
def index():
    """Main page"""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """API endpoint for price prediction"""
    try:
        # Get input data from form
        data = request.get_json()
        
        # Extract features
        avg_area_income = float(data['avg_area_income'])
        avg_area_house_age = float(data['avg_area_house_age'])
        avg_area_rooms = float(data['avg_area_rooms'])
        avg_area_bedrooms = float(data['avg_area_bedrooms'])
        area_population = float(data['area_population'])
        
        # Create feature array
        features = np.array([[
            avg_area_income,
            avg_area_house_age,
            avg_area_rooms,
            avg_area_bedrooms,
            area_population
        ]])
        
        # Make prediction
        prediction = model.predict(features)[0]
        
        # Format prediction
        formatted_prediction = f"${prediction:,.2f}"
        
        return jsonify({
            'success': True,
            'prediction': formatted_prediction,
            'raw_prediction': prediction
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })

@app.route('/model_info')
def model_info():
    """Get model information"""
    if model is None:
        return jsonify({'error': 'Model not loaded'})
    
    # Get feature coefficients
    coefficients = model.coef_
    intercept = model.intercept_
    
    feature_coeffs = []
    for i, feature in enumerate(feature_names):
        feature_coeffs.append({
            'feature': feature,
            'coefficient': coefficients[i]
        })
    
    return jsonify({
        'feature_coefficients': feature_coeffs,
        'intercept': intercept,
        'feature_names': feature_names
    })

if __name__ == '__main__':
    # Load and train model on startup
    print("Loading and training model...")
    model_result = load_and_train_model()
    
    if model_result[0] is None:
        print("Failed to load model. Exiting...")
        exit(1)
    
    # Run the app
    print("Starting Flask web application...")
    app.run(debug=True, host='0.0.0.0', port=5000)
