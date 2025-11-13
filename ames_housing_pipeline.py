"""
Complete Machine Learning Pipeline for Ames Housing Dataset
Includes: Data Processing, EDA, PyTorch MLP, XGBoost, Random Forest
With ONNX Export for Deployment
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import xgboost as xgb
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import warnings
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)

print("=" * 80)
print("AMES HOUSING PRICE PREDICTION - COMPLETE ML PIPELINE")
print("=" * 80)

# ============================================================================
# 1. DATA LOADING AND EXPLORATION
# ============================================================================
print("\n1. LOADING DATA...")

# Load the dataset
df = pd.read_csv('AmesHousing.csv')
print(f"Dataset shape: {df.shape}")
print(f"\nFirst few rows:")
print(df.head())

print(f"\nDataset Info:")
print(df.info())

print(f"\nBasic Statistics:")
print(df.describe())

print(f"\nMissing Values:")
missing = df.isnull().sum()
missing = missing[missing > 0].sort_values(ascending=False)
print(missing)

# ============================================================================
# 2. DATA PREPROCESSING
# ============================================================================
print("\n" + "=" * 80)
print("2. DATA PREPROCESSING")
print("=" * 80)

# Drop columns with too many missing values (>50%)
threshold = len(df) * 0.5
df = df.dropna(thresh=threshold, axis=1)

# Drop identifier columns
cols_to_drop = ['Order', 'PID']
df = df.drop(columns=[col for col in cols_to_drop if col in df.columns])

# Separate target variable
target_col = 'SalePrice'
y = df[target_col].copy()
X = df.drop(columns=[target_col])

print(f"\nFeatures shape: {X.shape}")
print(f"Target shape: {y.shape}")

# Identify numerical and categorical columns
numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
categorical_cols = X.select_dtypes(include=['object']).columns.tolist()

print(f"\nNumerical columns: {len(numerical_cols)}")
print(f"Categorical columns: {len(categorical_cols)}")

# Handle missing values
# For numerical: fill with median
for col in numerical_cols:
    if X[col].isnull().sum() > 0:
        X[col].fillna(X[col].median(), inplace=True)

# For categorical: fill with mode or 'None'
for col in categorical_cols:
    if X[col].isnull().sum() > 0:
        X[col].fillna(X[col].mode()[0] if len(X[col].mode()) > 0 else 'None', inplace=True)

# Encode categorical variables
label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col].astype(str))
    label_encoders[col] = le

print(f"\nAfter preprocessing - Features shape: {X.shape}")
print(f"Missing values remaining: {X.isnull().sum().sum()}")

# ============================================================================
# 3. EXPLORATORY DATA ANALYSIS
# ============================================================================
print("\n" + "=" * 80)
print("3. EXPLORATORY DATA ANALYSIS")
print("=" * 80)

# Target variable distribution
print(f"\nTarget Variable Statistics:")
print(f"Mean: ${y.mean():,.2f}")
print(f"Median: ${y.median():,.2f}")
print(f"Std: ${y.std():,.2f}")
print(f"Min: ${y.min():,.2f}")
print(f"Max: ${y.max():,.2f}")

# Feature correlation with target
correlations = X.apply(lambda x: x.corr(y)).sort_values(ascending=False)
print(f"\nTop 10 Features Correlated with SalePrice:")
print(correlations.head(10))

# ============================================================================
# 4. TRAIN-TEST SPLIT
# ============================================================================
print("\n" + "=" * 80)
print("4. SPLITTING DATA")
print("=" * 80)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"Training set: {X_train.shape[0]} samples")
print(f"Test set: {X_test.shape[0]} samples")

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ============================================================================
# 5. MODEL 1: RANDOM FOREST
# ============================================================================
print("\n" + "=" * 80)
print("5. RANDOM FOREST MODEL")
print("=" * 80)

print("\nTraining Random Forest...")
rf_model = RandomForestRegressor(
    n_estimators=100,
    max_depth=20,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1
)
rf_model.fit(X_train, y_train)

# Predictions
y_train_pred_rf = rf_model.predict(X_train)
y_test_pred_rf = rf_model.predict(X_test)

# Metrics
train_r2_rf = r2_score(y_train, y_train_pred_rf)
test_r2_rf = r2_score(y_test, y_test_pred_rf)
train_rmse_rf = np.sqrt(mean_squared_error(y_train, y_train_pred_rf))
test_rmse_rf = np.sqrt(mean_squared_error(y_test, y_test_pred_rf))
train_mae_rf = mean_absolute_error(y_train, y_train_pred_rf)
test_mae_rf = mean_absolute_error(y_test, y_test_pred_rf)

print(f"\nRandom Forest Results:")
print(f"Training R²: {train_r2_rf:.4f}")
print(f"Test R²: {test_r2_rf:.4f}")
print(f"Training RMSE: ${train_rmse_rf:,.2f}")
print(f"Test RMSE: ${test_rmse_rf:,.2f}")
print(f"Training MAE: ${train_mae_rf:,.2f}")
print(f"Test MAE: ${test_mae_rf:,.2f}")

# Feature importance
feature_importance_rf = pd.DataFrame({
    'feature': X.columns,
    'importance': rf_model.feature_importances_
}).sort_values('importance', ascending=False)

print(f"\nTop 10 Most Important Features:")
print(feature_importance_rf.head(10))

# ============================================================================
# 6. MODEL 2: XGBOOST
# ============================================================================
print("\n" + "=" * 80)
print("6. XGBOOST MODEL")
print("=" * 80)

print("\nTraining XGBoost...")
xgb_model = xgb.XGBRegressor(
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1
)
xgb_model.fit(X_train, y_train)

# Predictions
y_train_pred_xgb = xgb_model.predict(X_train)
y_test_pred_xgb = xgb_model.predict(X_test)

# Metrics
train_r2_xgb = r2_score(y_train, y_train_pred_xgb)
test_r2_xgb = r2_score(y_test, y_test_pred_xgb)
train_rmse_xgb = np.sqrt(mean_squared_error(y_train, y_train_pred_xgb))
test_rmse_xgb = np.sqrt(mean_squared_error(y_test, y_test_pred_xgb))
train_mae_xgb = mean_absolute_error(y_train, y_train_pred_xgb)
test_mae_xgb = mean_absolute_error(y_test, y_test_pred_xgb)

print(f"\nXGBoost Results:")
print(f"Training R²: {train_r2_xgb:.4f}")
print(f"Test R²: {test_r2_xgb:.4f}")
print(f"Training RMSE: ${train_rmse_xgb:,.2f}")
print(f"Test RMSE: ${test_rmse_xgb:,.2f}")
print(f"Training MAE: ${train_mae_xgb:,.2f}")
print(f"Test MAE: ${test_mae_xgb:,.2f}")

# ============================================================================
# 7. MODEL 3: PYTORCH MLP
# ============================================================================
print("\n" + "=" * 80)
print("7. PYTORCH MLP MODEL")
print("=" * 80)

# Custom Dataset
class HousingDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y.values).reshape(-1, 1)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# Create datasets and dataloaders
train_dataset = HousingDataset(X_train_scaled, y_train)
test_dataset = HousingDataset(X_test_scaled, y_test)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Define MLP Model
class MLPRegressor(nn.Module):
    def __init__(self, input_dim):
        super(MLPRegressor, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
    
    def forward(self, x):
        return self.network(x)

# Initialize model
input_dim = X_train_scaled.shape[1]
mlp_model = MLPRegressor(input_dim)
criterion = nn.MSELoss()
optimizer = optim.Adam(mlp_model.parameters(), lr=0.001, weight_decay=1e-5)

print(f"\nMLP Architecture:")
print(mlp_model)
print(f"\nTotal parameters: {sum(p.numel() for p in mlp_model.parameters())}")

# Training loop
print("\nTraining MLP...")
num_epochs = 100
train_losses = []
test_losses = []

for epoch in range(num_epochs):
    # Training
    mlp_model.train()
    train_loss = 0.0
    for X_batch, y_batch in train_loader:
        optimizer.zero_grad()
        outputs = mlp_model(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    
    train_loss /= len(train_loader)
    train_losses.append(train_loss)
    
    # Validation
    mlp_model.eval()
    test_loss = 0.0
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            outputs = mlp_model(X_batch)
            loss = criterion(outputs, y_batch)
            test_loss += loss.item()
    
    test_loss /= len(test_loader)
    test_losses.append(test_loss)
    
    if (epoch + 1) % 20 == 0:
        print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}")

# Predictions
mlp_model.eval()
with torch.no_grad():
    y_train_pred_mlp = mlp_model(torch.FloatTensor(X_train_scaled)).numpy().flatten()
    y_test_pred_mlp = mlp_model(torch.FloatTensor(X_test_scaled)).numpy().flatten()

# Metrics
train_r2_mlp = r2_score(y_train, y_train_pred_mlp)
test_r2_mlp = r2_score(y_test, y_test_pred_mlp)
train_rmse_mlp = np.sqrt(mean_squared_error(y_train, y_train_pred_mlp))
test_rmse_mlp = np.sqrt(mean_squared_error(y_test, y_test_pred_mlp))
train_mae_mlp = mean_absolute_error(y_train, y_train_pred_mlp)
test_mae_mlp = mean_absolute_error(y_test, y_test_pred_mlp)

print(f"\nPyTorch MLP Results:")
print(f"Training R²: {train_r2_mlp:.4f}")
print(f"Test R²: {test_r2_mlp:.4f}")
print(f"Training RMSE: ${train_rmse_mlp:,.2f}")
print(f"Test RMSE: ${test_rmse_mlp:,.2f}")
print(f"Training MAE: ${train_mae_mlp:,.2f}")
print(f"Test MAE: ${test_mae_mlp:,.2f}")

# ============================================================================
# 8. MODEL COMPARISON
# ============================================================================
print("\n" + "=" * 80)
print("8. MODEL COMPARISON")
print("=" * 80)

results = pd.DataFrame({
    'Model': ['Random Forest', 'XGBoost', 'PyTorch MLP'],
    'Train R²': [train_r2_rf, train_r2_xgb, train_r2_mlp],
    'Test R²': [test_r2_rf, test_r2_xgb, test_r2_mlp],
    'Train RMSE': [train_rmse_rf, train_rmse_xgb, train_rmse_mlp],
    'Test RMSE': [test_rmse_rf, test_rmse_xgb, test_rmse_mlp],
    'Train MAE': [train_mae_rf, train_mae_xgb, train_mae_mlp],
    'Test MAE': [test_mae_rf, test_mae_xgb, test_mae_mlp]
})

print("\nComparative Results:")
print(results.to_string(index=False))

# Find best model
best_model_idx = results['Test R²'].idxmax()
best_model_name = results.loc[best_model_idx, 'Model']
print(f"\n🏆 Best Model: {best_model_name} (Test R² = {results.loc[best_model_idx, 'Test R²']:.4f})")

# ============================================================================
# 9. ONNX EXPORT (PyTorch MLP)
# ============================================================================
print("\n" + "=" * 80)
print("9. EXPORTING PYTORCH MLP TO ONNX")
print("=" * 80)

try:
    import onnx
    import onnxruntime as ort
    
    # Export to ONNX
    dummy_input = torch.FloatTensor(X_test_scaled[:1])
    onnx_path = "mlp_housing_model.onnx"
    
    torch.onnx.export(
        mlp_model,
        dummy_input,
        onnx_path,
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
    )
    
    print(f"✓ Model exported to {onnx_path}")
    
    # Verify ONNX model
    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)
    print("✓ ONNX model verification successful")
    
    # Test ONNX inference
    ort_session = ort.InferenceSession(onnx_path)
    ort_inputs = {ort_session.get_inputs()[0].name: X_test_scaled[:5].astype(np.float32)}
    ort_outputs = ort_session.run(None, ort_inputs)
    
    print("✓ ONNX inference test successful")
    print(f"\nSample predictions (ONNX):")
    print(ort_outputs[0][:5].flatten())
    
except ImportError:
    print("⚠ ONNX libraries not installed. Install with: pip install onnx onnxruntime")

# ============================================================================
# 10. VISUALIZATIONS
# ============================================================================
print("\n" + "=" * 80)
print("10. GENERATING VISUALIZATIONS")
print("=" * 80)

# Create figure with subplots
fig = plt.figure(figsize=(20, 12))

# 1. Target distribution
ax1 = plt.subplot(3, 3, 1)
plt.hist(y, bins=50, edgecolor='black', alpha=0.7)
plt.xlabel('Sale Price ($)')
plt.ylabel('Frequency')
plt.title('Distribution of Sale Prices')
plt.grid(alpha=0.3)

# 2. Correlation heatmap (top features)
ax2 = plt.subplot(3, 3, 2)
top_features = correlations.head(10).index.tolist()
corr_matrix = X[top_features].corrwith(y).to_frame()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, cbar_kws={'label': 'Correlation'})
plt.title('Top 10 Feature Correlations with Price')
plt.tight_layout()

# 3. R² comparison
ax3 = plt.subplot(3, 3, 3)
models = results['Model']
train_r2 = results['Train R²']
test_r2 = results['Test R²']
x_pos = np.arange(len(models))
width = 0.35
plt.bar(x_pos - width/2, train_r2, width, label='Train R²', alpha=0.8)
plt.bar(x_pos + width/2, test_r2, width, label='Test R²', alpha=0.8)
plt.xlabel('Model')
plt.ylabel('R² Score')
plt.title('Model Performance Comparison (R²)')
plt.xticks(x_pos, models, rotation=15)
plt.legend()
plt.grid(axis='y', alpha=0.3)

# 4. RMSE comparison
ax4 = plt.subplot(3, 3, 4)
test_rmse = results['Test RMSE']
plt.bar(models, test_rmse, alpha=0.7, color=['#1f77b4', '#ff7f0e', '#2ca02c'])
plt.xlabel('Model')
plt.ylabel('RMSE ($)')
plt.title('Test RMSE Comparison')
plt.xticks(rotation=15)
plt.grid(axis='y', alpha=0.3)

# 5. Predicted vs Actual (XGBoost)
ax5 = plt.subplot(3, 3, 5)
plt.scatter(y_test, y_test_pred_xgb, alpha=0.5, s=20)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel('Actual Price ($)')
plt.ylabel('Predicted Price ($)')
plt.title(f'XGBoost: Predicted vs Actual (R²={test_r2_xgb:.4f})')
plt.grid(alpha=0.3)

# 6. Predicted vs Actual (Random Forest)
ax6 = plt.subplot(3, 3, 6)
plt.scatter(y_test, y_test_pred_rf, alpha=0.5, s=20)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel('Actual Price ($)')
plt.ylabel('Predicted Price ($)')
plt.title(f'Random Forest: Predicted vs Actual (R²={test_r2_rf:.4f})')
plt.grid(alpha=0.3)

# 7. Predicted vs Actual (MLP)
ax7 = plt.subplot(3, 3, 7)
plt.scatter(y_test, y_test_pred_mlp, alpha=0.5, s=20)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel('Actual Price ($)')
plt.ylabel('Predicted Price ($)')
plt.title(f'PyTorch MLP: Predicted vs Actual (R²={test_r2_mlp:.4f})')
plt.grid(alpha=0.3)

# 8. MLP training history
ax8 = plt.subplot(3, 3, 8)
plt.plot(train_losses, label='Train Loss', linewidth=2)
plt.plot(test_losses, label='Test Loss', linewidth=2)
plt.xlabel('Epoch')
plt.ylabel('MSE Loss')
plt.title('MLP Training History')
plt.legend()
plt.grid(alpha=0.3)

# 9. Feature importance (top 15)
ax9 = plt.subplot(3, 3, 9)
top_15_features = feature_importance_rf.head(15)
plt.barh(range(len(top_15_features)), top_15_features['importance'])
plt.yticks(range(len(top_15_features)), top_15_features['feature'])
plt.xlabel('Importance')
plt.title('Top 15 Feature Importances (Random Forest)')
plt.gca().invert_yaxis()
plt.grid(axis='x', alpha=0.3)

plt.tight_layout()
plt.savefig('ames_housing_analysis.png', dpi=300, bbox_inches='tight')
print("✓ Visualizations saved to 'ames_housing_analysis.png'")

plt.show()

# ============================================================================
# 11. FINAL SUMMARY
# ============================================================================
print("\n" + "=" * 80)
print("11. FINAL SUMMARY")
print("=" * 80)

print(f"""
Analysis Complete!

Dataset: Ames Housing
- Total samples: {len(df)}
- Features: {X.shape[1]}
- Target: Sale Price

Best Performing Model: {best_model_name}
- Test R²: {results.loc[best_model_idx, 'Test R²']:.4f}
- Test RMSE: ${results.loc[best_model_idx, 'Test RMSE']:,.2f}
- Test MAE: ${results.loc[best_model_idx, 'Test MAE']:,.2f}

All models have been trained and evaluated.
PyTorch MLP model exported to ONNX format for deployment.

Files generated:
- mlp_housing_model.onnx (ONNX model for deployment)
- ames_housing_analysis.png (visualizations)
""")

print("=" * 80)
print("PIPELINE COMPLETE!")
print("=" * 80)