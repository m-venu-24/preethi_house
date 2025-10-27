import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor

# Function to load and preprocess data
def load_and_preprocess_data(file_path):
    print("Loading and preprocessing data...")
    # Load the data
    df = pd.read_csv(file_path)
    
    # Display basic information
    print("\nData Overview:")
    print(f"Shape: {df.shape}")
    print("\nData Types:")
    print(df.dtypes)
    
    # Check for missing values
    print("\nMissing Values:")
    print(df.isnull().sum())
    
    # Check for duplicates
    duplicates = df.duplicated().sum()
    print(f"\nNumber of duplicates: {duplicates}")
    
    # Drop address column if it exists
    if 'address' in df.columns:
        df = df.drop('address', axis=1)
        print("\nDropped 'address' column as it's not significant for analysis")
    elif 'Address' in df.columns:
        df = df.drop('Address', axis=1)
        print("\nDropped 'Address' column as it's not significant for analysis")
    
    return df

# Function to perform exploratory data analysis
def perform_eda(df):
    print("\nPerforming Exploratory Data Analysis...")
    
    # Summary statistics
    print("\nSummary Statistics:")
    print(df.describe())
    
    # Correlation matrix
    plt.figure(figsize=(12, 10))
    correlation_matrix = df.corr()
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
    plt.title('Correlation Matrix')
    plt.savefig('data/correlation_matrix.png')
    print("Correlation matrix saved to 'data/correlation_matrix.png'")
    
    # Distribution of the target variable (price)
    plt.figure(figsize=(10, 6))
    price_col = 'Price' if 'Price' in df.columns else 'price'
    sns.histplot(df[price_col], kde=True)
    plt.title('Distribution of House Prices')
    plt.xlabel('Price')
    plt.savefig('data/price_distribution.png')
    print("Price distribution plot saved to 'data/price_distribution.png'")
    
    return correlation_matrix

# Function to detect and remove outliers using IQR method
def remove_outliers_iqr(df, column='Price'):
    print("\nDetecting and removing outliers using IQR method...")
    
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)]
    print(f"Number of outliers detected: {len(outliers)}")
    
    # Remove outliers
    df_clean = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
    print(f"Shape after removing outliers: {df_clean.shape}")
    
    return df_clean

# Function to check for multicollinearity
def check_multicollinearity(X):
    print("\nChecking for multicollinearity...")
    
    # Add constant for statsmodels
    X_with_const = sm.add_constant(X)
    
    # Calculate VIF for each feature
    vif_data = pd.DataFrame()
    vif_data["Feature"] = X_with_const.columns
    vif_data["VIF"] = [variance_inflation_factor(X_with_const.values, i) for i in range(X_with_const.shape[1])]
    
    print(vif_data)
    print("VIF values less than 5 indicate no significant multicollinearity")
    
    return vif_data

# Function to run linear regression with statsmodels
def run_linear_regression_statsmodels(X, y):
    print("\nRunning Linear Regression with statsmodels...")
    
    # Add constant for statsmodels
    X_with_const = sm.add_constant(X)
    
    # Fit the model
    model = sm.OLS(y, X_with_const).fit()
    
    # Print summary
    print(model.summary())
    
    return model

# Function to split data and train model
def train_test_model(X, y, test_size=0.2, random_state=42):
    print("\nSplitting data into train and test sets...")
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    print(f"Training set shape: X_train {X_train.shape}, y_train {y_train.shape}")
    print(f"Testing set shape: X_test {X_test.shape}, y_test {y_test.shape}")
    
    # Train the model using sklearn
    print("\nTraining Linear Regression model...")
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # Print coefficients
    print("\nModel Coefficients:")
    coef_df = pd.DataFrame({
        'Feature': X.columns,
        'Coefficient': model.coef_
    })
    print(coef_df)
    print(f"Intercept: {model.intercept_}")
    
    # Make predictions
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    # Calculate metrics
    train_r2 = model.score(X_train, y_train)
    test_r2 = model.score(X_test, y_test)
    
    train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
    
    train_mape = mean_absolute_percentage_error(y_train, y_train_pred)
    test_mape = mean_absolute_percentage_error(y_test, y_test_pred)
    
    print("\nModel Performance:")
    print(f"Training R² Score: {train_r2:.4f}")
    print(f"Testing R² Score: {test_r2:.4f}")
    print(f"Training RMSE: {train_rmse:.2f}")
    print(f"Testing RMSE: {test_rmse:.2f}")
    print(f"Training MAPE: {train_mape:.4f}")
    print(f"Testing MAPE: {test_mape:.4f}")
    print(f"Model Accuracy: {(1 - test_mape) * 100:.2f}%")
    
    # Visualize predictions vs actual
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_test_pred, alpha=0.5)
    plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--')
    plt.xlabel('Actual Prices')
    plt.ylabel('Predicted Prices')
    plt.title('Actual vs Predicted House Prices')
    plt.savefig('data/actual_vs_predicted.png')
    print("Actual vs Predicted plot saved to 'data/actual_vs_predicted.png'")
    
    return model, X_train, X_test, y_train, y_test, y_train_pred, y_test_pred

# Main function
def main():
    # Load real house data
    print("Loading real house price data...")
    
    # Load and preprocess data
    df = load_and_preprocess_data('real_house_data.csv')
    
    # Perform EDA
    correlation_matrix = perform_eda(df)
    
    # Prepare features and target
    price_col = 'Price' if 'Price' in df.columns else 'price'
    X = df.drop(price_col, axis=1)
    y = df[price_col]
    
    # Run initial linear regression
    initial_model = run_linear_regression_statsmodels(X, y)
    print(f"\nInitial Multiple R-squared: {initial_model.rsquared:.4f}")
    
    # Remove outliers
    df_clean = remove_outliers_iqr(df)
    
    # Prepare features and target after outlier removal
    X_clean = df_clean.drop(price_col, axis=1)
    y_clean = df_clean[price_col]
    
    # Check for multicollinearity
    vif_data = check_multicollinearity(X_clean)
    
    # Run linear regression after outlier removal
    clean_model = run_linear_regression_statsmodels(X_clean, y_clean)
    print(f"\nMultiple R-squared after outlier removal: {clean_model.rsquared:.4f}")
    
    # Split data and train model
    model, X_train, X_test, y_train, y_test, y_train_pred, y_test_pred = train_test_model(X_clean, y_clean)
    
    print("\nHouse Price Prediction Model Complete!")
    print("Analysis based on real house price data with features:")
    print("- Avg. Area Income")
    print("- Avg. Area House Age") 
    print("- Avg. Area Number of Rooms")
    print("- Avg. Area Number of Bedrooms")
    print("- Area Population")
    print(f"The model shows strong predictive power with R² = {clean_model.rsquared:.3f}")
    print(f"Model accuracy: {(1 - model.score(X_test, y_test)) * 100:.2f}% based on R²")

if __name__ == "__main__":
    main()