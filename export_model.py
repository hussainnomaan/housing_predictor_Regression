import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
import torch
import torch.nn as nn
import torch.onnx

# --------------------------
# Load Data
# --------------------------
print("Loading data...")
data = pd.read_csv("AmesHousing.csv")  # Replace with your CSV

# --------------------------
# Feature Selection & Preprocessing
# --------------------------
features = ['Overall Qual', 'Gr Liv Area', 'Garage Cars', 'Garage Area',
    'Total Bsmt SF', '1st Flr SF', 'Year Built', 'Full Bath',
    'TotRms AbvGrd', 'Lot Area']
target = 'SalePrice'

# Make sure all features exist in the CSV
data = data[features + [target]].copy()

# Handle missing values
data.fillna(0, inplace=True)

X = data[features]
y = data[target]

# Scale numeric features for MLP
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y.values, test_size=0.2, random_state=42)

# --------------------------
# Train Random Forest
# --------------------------
print("Training Random Forest...")
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Export RF to ONNX
try:
    import skl2onnx
    from skl2onnx import convert_sklearn
    from skl2onnx.common.data_types import FloatTensorType

    print("Exporting RF to ONNX...")
    initial_type = [('float_input', FloatTensorType([None, X_train.shape[1]]))]
    rf_onnx = convert_sklearn(rf_model, initial_types=initial_type)
    with open("rf_housing_model.onnx", "wb") as f:
        f.write(rf_onnx.SerializeToString())
    print("Random Forest exported to rf_housing_model.onnx ✅")
except Exception as e:
    print("Failed to export RF:", e)

# --------------------------
# Train PyTorch MLP
# --------------------------
print("Training MLP...")

class MLP(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        return self.layers(x)

mlp_model = MLP(X_train.shape[1])
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(mlp_model.parameters(), lr=0.001)

X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train.reshape(-1, 1), dtype=torch.float32)

# Training loop
epochs = 200
for epoch in range(epochs):
    optimizer.zero_grad()
    outputs = mlp_model(X_train_tensor)
    loss = criterion(outputs, y_train_tensor)
    loss.backward()
    optimizer.step()
    if (epoch+1) % 50 == 0:
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.2f}")

# Export MLP to ONNX
print("Exporting MLP to ONNX (browser-safe)...")
dummy_input = torch.randn(1, X_train.shape[1])

torch.onnx.export(
    mlp_model,
    dummy_input,
    "mlp_housing_model.onnx",
    input_names=["float_input"],
    output_names=["output"],
    opset_version=18,
    export_params=True,
    do_constant_folding=True,
    keep_initializers_as_inputs=False,   # ← CRITICAL: prevents .data file
    dynamic_axes=None
)

print("MLP exported as SINGLE .onnx file (no .data) → ready for GitHub Pages")