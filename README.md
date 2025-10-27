# House Price Prediction Model

This project implements a linear regression model to predict house prices based on various features such as area, number of bedrooms, bathrooms, age of the house, and garage size.

## Project Overview

The model follows these key steps:
1. Data loading and preprocessing
2. Exploratory data analysis (EDA)
3. Initial linear regression model
4. Outlier detection and removal using IQR method
5. Multicollinearity check
6. Train-test split and final model training
7. Model evaluation using RMSE and MAPE metrics

## Key Findings

- The model achieves an R-squared value of 0.921, indicating that the independent variables explain 92.1% of the variation in house prices
- RMSE (Root Mean Squared Error) is approximately 99,886.24
- MAPE (Mean Absolute Percentage Error) is 0.07, indicating a model accuracy of 92.45%
- All features except for "AreaNumberofBedrooms" are statistically significant (p-value < alpha)
- No significant multicollinearity was detected between independent variables (all VIF values < 5)
- With an increase of 1 bedroom, the house price increases by approximately $118,602

## Requirements

- Python 3.6+
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn
- statsmodels

## Usage

Run the main script:

```
python house_price_prediction.py
```

This will:
1. Generate a sample dataset (if no data is provided)
2. Perform all analysis steps
3. Output model performance metrics
4. Save visualization plots to the data directory

## Visualizations

The script generates several visualizations:
- Correlation matrix heatmap
- Distribution of house prices
- Actual vs. predicted prices scatter plot

These are saved in the `data` directory for reference.