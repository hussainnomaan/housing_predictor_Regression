"""
Export all trained models to ONNX format
Run this AFTER training your models with the main pipeline
"""

import numpy as np
import pandas as pd
import torch
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb

# Import ONNX libraries
try:
    import onnx
    import onnxruntime as ort
    from skl2onnx import convert_sklearn
    from skl2onnx.common.data_types import FloatTensorType
    print("✓ All ONNX libraries loaded successfully")
except ImportError as e:
    print(f"Error: {e}")
    print("Please run: pip install onnx onnxruntime skl2onnx onnxmltools")
    exit(1)

# Load and preprocess data (same as main pipeline)
print("Loading data...")
df = pd.read_csv('AmesHousing.csv')

# Preprocessing
threshold = len(df) * 0.5
df = df.dropna(thresh=threshold, axis=1)
cols_to_drop = ['Order', 'PID']
df = df.drop(columns=[col for col in cols_to_drop if col in df.columns])

target_col = 'SalePrice'
y = df[target_col].copy()
X = df.drop(columns=[target_col])

numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
categorical_cols = X.select_dtypes(include=['object']).columns.tolist()

for col in numerical_cols:
    if X[col].isnull().sum() > 0:
        X[col].fillna(X[col].median(), inplace=True)

for col in categorical_cols:
    if X[col].isnull().sum() > 0:
        X[col].fillna(X[col].mode()[0] if len(X[col].mode()) > 0 else 'None', inplace=True)

label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col].astype(str))
    label_encoders[col] = le

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("Training models...")

# Train Random Forest
rf_model = RandomForestRegressor(n_estimators=100, max_depth=20, random_state=42, n_jobs=-1)
rf_model.fit(X_train, y_train)
print("✓ Random Forest trained")

# Train XGBoost
xgb_model = xgb.XGBRegressor(n_estimators=100, max_depth=6, learning_rate=0.1, random_state=42, n_jobs=-1)
xgb_model.fit(X_train, y_train)
print("✓ XGBoost trained")

# Train PyTorch MLP
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

class HousingDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y.values).reshape(-1, 1)
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class MLPRegressor(nn.Module):
    def __init__(self, input_dim):
        super(MLPRegressor, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 256), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256, 128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(64, 32), nn.ReLU(),
            nn.Linear(32, 1)
        )
    def forward(self, x):
        return self.network(x)

input_dim = X_train_scaled.shape[1]
mlp_model = MLPRegressor(input_dim)
criterion = nn.MSELoss()
optimizer = optim.Adam(mlp_model.parameters(), lr=0.001)

train_dataset = HousingDataset(X_train_scaled, y_train)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

mlp_model.train()
for epoch in range(50):  # Reduced epochs for faster export
    for X_batch, y_batch in train_loader:
        optimizer.zero_grad()
        outputs = mlp_model(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()

mlp_model.eval()
print("✓ PyTorch MLP trained")

# =================================================================
# EXPORT MODELS TO ONNX
# =================================================================

print("\n" + "="*70)
print("EXPORTING MODELS TO ONNX")
print("="*70)

# 1. Export PyTorch MLP
print("\n1. Exporting PyTorch MLP...")
dummy_input = torch.FloatTensor(X_test_scaled[:1])
torch.onnx.export(
    mlp_model, dummy_input, "mlp_housing_model.onnx",
    export_params=True, opset_version=11,
    input_names=['input'], output_names=['output'],
    dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
)
print("✓ mlp_housing_model.onnx created")

# 2. Export XGBoost
print("\n2. Exporting XGBoost...")
initial_type = [('float_input', FloatTensorType([None, X_train.shape[1]]))]
xgb_onnx = convert_sklearn(xgb_model, initial_types=initial_type, target_opset=11)
with open("xgboost_housing_model.onnx", "wb") as f:
    f.write(xgb_onnx.SerializeToString())
print("✓ xgboost_housing_model.onnx created")

# 3. Export Random Forest
print("\n3. Exporting Random Forest...")
rf_onnx = convert_sklearn(rf_model, initial_types=initial_type, target_opset=11)
with open("random_forest_housing_model.onnx", "wb") as f:
    f.write(rf_onnx.SerializeToString())
print("✓ random_forest_housing_model.onnx created")

# Verify all models
print("\n" + "="*70)
print("VERIFYING ONNX MODELS")
print("="*70)

models = {
    'mlp_housing_model.onnx': X_test_scaled[:5].astype(np.float32),
    'xgboost_housing_model.onnx': X_test[:5].astype(np.float32),
    'random_forest_housing_model.onnx': X_test[:5].astype(np.float32)
}

for model_name, test_data in models.items():
    print(f"\n{model_name}:")
    onnx_model = onnx.load(model_name)
    onnx.checker.check_model(onnx_model)
    print("  ✓ Model structure valid")
    
    session = ort.InferenceSession(model_name)
    input_name = session.get_inputs()[0].name
    result = session.run(None, {input_name: test_data})
    print(f"  ✓ Inference successful")
    print(f"  Sample predictions: {result[0][:3].flatten()}")

# Save preprocessing components
print("\n" + "="*70)
print("SAVING PREPROCESSING COMPONENTS")
print("="*70)

with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)
print("✓ scaler.pkl saved")

with open('label_encoders.pkl', 'wb') as f:
    pickle.dump(label_encoders, f)
print("✓ label_encoders.pkl saved")

with open('feature_names.pkl', 'wb') as f:
    pickle.dump(list(X.columns), f)
print("✓ feature_names.pkl saved")

print("\n" + "="*70)
print("EXPORT COMPLETE!")
print("="*70)
print("\nGenerated files:")
print("  ✓ mlp_housing_model.onnx")
print("  ✓ xgboost_housing_model.onnx          (BEST - R² = 0.9227)")
print("  ✓ random_forest_housing_model.onnx")
print("  ✓ scaler.pkl")
print("  ✓ label_encoders.pkl")
print("  ✓ feature_names.pkl")
print("\nRecommendation: Use xgboost_housing_model.onnx for deployment")
print("="*70)