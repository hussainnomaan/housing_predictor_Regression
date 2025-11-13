# run_xgboost.py
# This is the final, working script for the XGBoost model.
# It calculates metrics and embeds the model into index.html.

import pandas as pd
import numpy as np
import xgboost as xgb
import base64
from pathlib import Path
import re
import warnings

# Imports for ONNX conversion
import onnxmltools
from onnxmltools.convert.common.data_types import FloatTensorType

# Imports for METRICS
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# Suppress warnings
warnings.filterwarnings('ignore')

# Load data
print("Loading data...")
data = pd.read_csv("AmesHousing.csv")
features = ['Overall Qual', 'Gr Liv Area', 'Garage Cars', 'Garage Area',
            'Total Bsmt SF', '1st Flr SF', 'Year Built', 'Full Bath',
            'TotRms AbvGrd', 'Lot Area']
target = 'SalePrice'

data = data[features + [target]].copy()
data.fillna(0, inplace=True)

X = data[features].values
y = data[target].values

# ============================================================================
# 1. MODEL EVALUATION (Calculate Metrics)
# ============================================================================
print("\n" + "=" * 80)
print("1. EVALUATING XGBOOST MODEL PERFORMANCE...")
print("=" * 80)

# Split data into training and testing sets FOR EVALUATION
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Training set size: {X_train.shape[0]}")
print(f"Test set size: {X_test.shape[0]}")

# Train an evaluation model ONLY on the training data
print("\nTraining evaluation XGBoost model on 80% of data...")
eval_model = xgb.XGBRegressor(n_estimators=500, max_depth=8, learning_rate=0.05, random_state=42)
eval_model.fit(X_train, y_train)

# Make predictions on the unseen test data
print("Making predictions on 20% test data...")
y_pred = eval_model.predict(X_test)

# Calculate metrics
r2 = r2_score(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae = mean_absolute_error(y_test, y_pred)

print("\n--- XGBOOST MODEL METRICS (on unseen test data) ---")
print(f"R² (R-squared):   {r2:.4f}")
print(f"RMSE (Root MSE):  ${rmse:,.2f}")
print(f"MAE (Mean Abs Err): ${mae:,.2f}")
print("-----------------------------------------------")


# ============================================================================
# 2. FINAL MODEL TRAINING (for Deployment)
# ============================================================================
print("\n" + "=" * 80)
print("2. TRAINING FINAL XGBOOST MODEL FOR DEPLOYMENT...")
print("=" * 80)

# Train the final model on 100% of the data
print("Training final model on 100% of data...")
model = xgb.XGBRegressor(n_estimators=500, max_depth=8, learning_rate=0.05, random_state=42)
model.fit(X, y)

print("Low-end house:", model.predict(np.array([[5,1000,1,300,500,500,1950,1,5,5000]]))[0])
print("High-end house:", model.predict(np.array([[9,3000,3,800,1500,1500,2020,3,10,15000]]))[0])

# --- CORRECTED ONNX EXPORT ---
print("\nConverting XGBoost to ONNX...")
onnx_path = "housing_model.onnx"

initial_types = [('input', FloatTensorType([1, 10]))]

# Convert the model (using the fix that works with older onnxmltools)
onnx_model = onnxmltools.convert_xgboost(
    model,
    initial_types=initial_types,
    target_opset=11
)

with open(onnx_path, "wb") as f:
    f.write(onnx_model.SerializeToString())

print(f"Exported → {onnx_path}")
# --- END OF CORRECTION ---

# ============================================================================
# 3. EMBEDDING MODEL IN HTML
# ============================================================================
print("\nEmbedding model into HTML...")
b64 = base64.b64encode(open(onnx_path, "rb").read()).decode()

# Read your HTML
with open("index.html", "r", encoding="utf-8") as f:
    html = f.read()

# This script injects your new metrics (R2, RMSE, MAE) into the UI
new_script = f"""<script>
let session = null;

async function loadModel() {{
    const status = document.getElementById('status');
    status.className = 'status info';
    status.textContent = 'Loading XGBoost model...';
    status.style.display = 'block';

    try {{
        session = await ort.InferenceSession.create('data:model/octet-stream;base64,{b64}');
        status.className = 'status success';
        status.textContent = 'XGBoost loaded!';
        document.getElementById('badge').textContent = 'XGBoost Model';
        
        // --- Metrics are now injected from your Python script ---
        document.getElementById('metricR2').textContent = '{r2:.4f}';
        document.getElementById('metricRMSE').textContent = '${rmse:,.0f}';
        document.getElementById('metricMAE').textContent = '${mae:,.0f}';
        // --------------------------------------------------------

        document.getElementById('metricModel').textContent = 'XGBoost';
        setTimeout(() => status.style.display = 'none', 3000);
    }} catch (e) {{
        status.className = 'status error';
        status.textContent = 'Load failed';
        console.error(e);
    }}
}}

async function predictPrice() {{
    if (!session) {{
        document.getElementById('status').textContent = 'Model not loaded';
        return;
    }}

    const status = document.getElementById('status');
    status.className = 'status info';
    status.textContent = 'Predicting...';
    status.style.display = 'block';

    try {{
        const values = [
            +document.getElementById('overallQual').value,
            +document.getElementById('grLivArea').value,
            +document.getElementById('garageCars').value,
            +document.getElementById('garageArea').value,
            +document.getElementById('totalBsmtSF').value,
            +document.getElementById('firstFlrSF').value,
            +document.getElementById('yearBuilt').value,
            +document.getElementById('fullBath').value,
            +document.getElementById('totRmsAbvGrd').value,
            +document.getElementById('lotArea').value
        ];
        
        const input = new ort.Tensor('float32', new Float32Array(values), [1, 10]);
        const output = await session.run({{input: input}});
        
        // --- FIX for default output name ---
        const price = output.variable.data[0];
        // ---------------------------------

        document.getElementById('predictedPrice').textContent = '$' + Math.round(price).toLocaleString();
        document.getElementById('modelUsed').textContent = 'Using: XGBoost Model';
        document.getElementById('result').classList.add('show');

        status.className = 'status success';
        status.textContent = 'Done!';
        setTimeout(() => status.style.display = 'none', 2000);
    }} catch (e) {{
        status.className = 'status error';
        status.textContent = 'Error';
        console.error(e);
    }}
}}

window.addEventListener('load', loadModel);
</script>"""

# --- Inject new HTML and JavaScript into the file ---

# First, remove any old comparison table (if the other script half-ran)
html = re.sub(r'<div class="comparison-table">.*?</div>', '', html, flags=re.DOTALL)

# Second, replace the old script tag
html = re.sub(r'<script>.*</script>', new_script, html, flags=re.DOTALL)

# Write the updated HTML back to the file
Path("index.html").write_text(html, encoding="utf-8")

print("\nYOUR PROJECT IS NOW 100% WORKING (WITH XGBOOST)")
print("Metrics have been calculated and printed to the console.")
print("The new metrics have also been embedded into the index.html file.")