# check_metrics.py
# This script only trains a LightGBM model to evaluate its performance metrics.

import pandas as pd
import numpy as np
import lightgbm as lgb

# Imports for METRICS
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# ============================================================================
# 1. LOAD AND PREPARE DATA
# ============================================================================
print("Loading data...")
data = pd.read_csv("AmesHousing.csv")

# Define the 10 features used in your frontend
features = ['Overall Qual', 'Gr Liv Area', 'Garage Cars', 'Garage Area',
            'Total Bsmt SF', '1st Flr SF', 'Year Built', 'Full Bath',
            'TotRms AbvGrd', 'Lot Area']
target = 'SalePrice'

data = data[features + [target]].copy()
data.fillna(0, inplace=True)

X = data[features].values
y = data[target].values

# ============================================================================
# 2. MODEL EVALUATION (Calculate Metrics)
# ============================================================================
print("\n" + "=" * 80)
print("EVALUATING MODEL PERFORMANCE...")
print("=" * 80)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Training set size: {X_train.shape[0]}")
print(f"Test set size: {X_test.shape[0]}")

# Train the model ONLY on the training data
print("\nTraining evaluation model on 80% of data...")
eval_model = lgb.LGBMRegressor(n_estimators=500, max_depth=8, learning_rate=0.05, random_state=42)
eval_model.fit(X_train, y_train)

# Make predictions on the unseen test data
print("Making predictions on 20% test data...")
y_pred = eval_model.predict(X_test)

# Calculate metrics
r2 = r2_score(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae = mean_absolute_error(y_test, y_pred)

print("\n" + "=" * 80)
print("--- MODEL METRICS (on unseen test data) ---")
print(f"R² (R-squared):   {r2:.4f}")
print(f"RMSE (Root MSE):  ${rmse:,.2f}")
print(f"MAE (Mean Abs Err): ${mae:,.2f}")
print("=" * 80)