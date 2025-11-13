# build_frontend.py
# This script trains Linear, MLP, XGBoost, and RF models,
# compares them, and embeds the BEST model (or pipeline) into index.html

import pandas as pd
import numpy as np
import base64
from pathlib import Path
import re
import warnings

# --- Model Imports ---
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
import lightgbm as lgb

# --- Pipeline & Metrics Imports ---
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# --- ONNX Imports ---
import onnxmltools
from onnxmltools.convert.common.data_types import FloatTensorType
import onnxconverter_common

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# ============================================================================
# 1. LOAD AND PREPARE DATA
# ============================================================================
print("Loading data...")
data = pd.read_csv("AmesHousing.csv")

features = ['Overall Qual', 'Gr Liv Area', 'Garage Cars', 'Garage Area',
            'Total Bsmt SF', '1st Flr SF', 'Year Built', 'Full Bath',
            'TotRms AbvGrd', 'Lot Area']
target = 'SalePrice'

data = data[features + [target]].copy()
data.fillna(0, inplace=True)

# We need both scaled and unscaled data
X_raw = data[features]
y_raw = data[target].values

# Split data for evaluation
X_train_raw, X_test_raw, y_train, y_test = train_test_split(X_raw, y_raw, test_size=0.2, random_state=42)

# Create scaled data just for the models that need it
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_raw)
X_test_scaled = scaler.transform(X_test_raw)

print(f"Training set size: {X_train_raw.shape[0]}")
print(f"Test set size: {X_test_raw.shape[0]}")

# ============================================================================
# 2. DEFINE, TRAIN, & EVALUATE MODELS
# ============================================================================
print("\n" + "=" * 80)
print("1. EVALUATING ALL MODELS...")
print("=" * 80)

# --- Define Models ---
# We define them separately so we can train them on different data (raw vs scaled)
models_to_evaluate = {
    "Linear Regression": LinearRegression(),
    "MLP Regressor": MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=500, random_state=42, early_stopping=True),
    "Random Forest": RandomForestRegressor(n_estimators=100, max_depth=20, random_state=42, n_jobs=-1),
    "XGBoost": xgb.XGBRegressor(n_estimators=500, max_depth=8, learning_rate=0.05, random_state=42),
    "LightGBM": lgb.LGBMRegressor(n_estimators=500, max_depth=8, learning_rate=0.05, random_state=42, verbose=-1)
}

all_metrics = {}
best_model_name = ""
best_model_object = None # This will be the model OR the pipeline
best_r2 = -float('inf')
best_model_metrics = {}
best_model_is_pipeline = False

# --- Loop and evaluate ---
for name, model in models_to_evaluate.items():
    print(f"\nTraining {name}...")
    
    # Tree models (RF, XGB, LGBM) are trained on RAW data
    if name in ["Random Forest", "XGBoost", "LightGBM"]:
        model.fit(X_train_raw, y_train)
        y_pred = model.predict(X_test_raw)
        
    # Linear/MLP models are trained on SCALED data
    else:
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
    
    # Calculate metrics
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    
    metrics = {"r2": r2, "rmse": rmse, "mae": mae}
    all_metrics[name] = metrics
    
    print(f"--- {name} Metrics ---")
    print(f"  R²:   {r2:.4f}")
    print(f"  RMSE: ${rmse:,.2f}")
    print(f"  MAE:  ${mae:,.2f}")
    
    if r2 > best_r2:
        best_r2 = r2
        best_model_name = name
        best_model_metrics = metrics
        best_model_object = model # Store the actual trained model object

print("\n" + "=" * 80)
print(f"🏆 BEST MODEL: {best_model_name} (Test R² = {best_r2:.4f})")
print("=" * 80)

# ============================================================================
# 3. FINAL MODEL EXPORT (Best Model Only)
# ============================================================================
print(f"\nTraining final {best_model_name} model on 100% of data for deployment...")
onnx_path = "housing_model.onnx"
initial_types = [('input', FloatTensorType([1, len(features)]))]

# If the best model is a tree, train on raw data and export *just the model*
if best_model_name in ["Random Forest", "XGBoost", "LightGBM"]:
    final_model = models_to_evaluate[best_model_name]
    final_model.fit(X_raw, y_raw) # Train on 100% RAW data
    
    print(f"Converting {best_model_name} (model only) to ONNX...")
    if best_model_name == "LightGBM":
        onnx_model = onnxmltools.convert_lightgbm(final_model, initial_types=initial_types, target_opset=11)
    elif best_model_name == "XGBoost":
        onnx_model = onnxmltools.convert_xgboost(final_model, initial_types=initial_types, target_opset=11)
    elif best_model_name == "Random Forest":
        onnx_model = onnxmltools.convert_sklearn(final_model, initial_types=initial_types, target_opset=11)

# If the best model is Linear/MLP, train on scaled data and export the *full pipeline*
else:
    print(f"Creating and training final {best_model_name} pipeline on 100% of data...")
    # 1. Create the final scaler
    final_scaler = StandardScaler()
    final_scaler.fit(X_raw) # Fit scaler on 100% RAW data
    
    # 2. Create the final model
    final_model = models_to_evaluate[best_model_name]
    X_scaled = final_scaler.transform(X_raw) # Scale all data
    final_model.fit(X_scaled, y_raw) # Train on 100% SCALED data
    
    # 3. Create the final pipeline
    final_pipeline = Pipeline([
        ('scaler', final_scaler),
        ('model', final_model)
    ])
    
    print(f"Converting {best_model_name} (full pipeline) to ONNX...")
    onnx_model = onnxmltools.convert_sklearn(
        final_pipeline,
        initial_types=initial_types,
        target_opset=11
    )

# Save the final ONNX model
with open(onnx_path, "wb") as f:
    f.write(onnx_model.SerializeToString())

print(f"Exported → {onnx_path}")


# ============================================================================
# 4. EMBEDDING MODEL AND ALL METRICS IN HTML
# ============================================================================
print("\nEmbedding model and all metrics into HTML...")
b64 = base64.b64encode(open(onnx_path, "rb").read()).decode()

# Read your HTML
with open("index.html", "r", encoding="utf-8") as f:
    html = f.read()

# --- 1. Create the new HTML Comparison Table ---
comparison_table_html = f"""
<div class="comparison-table">
  <h3 style="text-align: center; color: #333; margin-bottom: 15px;">Model Comparison (Test Set)</h3>
  <table>
    <thead>
      <tr>
        <th>Model</th>
        <th>Test R²</th>
        <th>Test RMSE</th>
        <th>Test MAE</th>
      </tr>
    </thead>
    <tbody>
      <tr>
        <td><b>LightGBM</b></td>
        <td>{all_metrics['LightGBM']['r2']:.4f}</td>
        <td>${all_metrics['LightGBM']['rmse']:,.0f}</td>
        <td>${all_metrics['LightGBM']['mae']:,.0f}</td>
      </tr>
      <tr>
        <td><b>XGBoost</b></td>
        <td>{all_metrics['XGBoost']['r2']:.4f}</td>
        <td>${all_metrics['XGBoost']['rmse']:,.0f}</td>
        <td>${all_metrics['XGBoost']['mae']:,.0f}</td>
      </tr>
      <tr>
        <td><b>Random Forest</b></td>
        <td>{all_metrics['Random Forest']['r2']:.4f}</td>
        <td>${all_metrics['Random Forest']['rmse']:,.0f}</td>
        <td>${all_metrics['Random Forest']['mae']:,.0f}</td>
      </tr>
      <tr>
        <td><b>MLP Regressor</b></td>
        <td>{all_metrics['MLP Regressor']['r2']:.4f}</td>
        <td>${all_metrics['MLP Regressor']['rmse']:,.0f}</td>
        <td>${all_metrics['MLP Regressor']['mae']:,.0f}</td>
      </tr>
      <tr>
        <td><b>Linear Regression</b></td>
        <td>{all_metrics['Linear Regression']['r2']:.4f}</td>
        <td>${all_metrics['Linear Regression']['rmse']:,.0f}</td>
        <td>${all_metrics['Linear Regression']['mae']:,.0f}</td>
      </tr>
    </tbody>
  </table>
</div>
"""

# --- 2. Create the new JavaScript ---
# It loads the BEST model and populates the main cards with ITS metrics
new_script = f"""<script>
let session = null;

async function loadModel() {{
    const status = document.getElementById('status');
    status.className = 'status info';
    status.textContent = 'Loading {best_model_name} model...';
    status.style.display = 'block';

    try {{
        session = await ort.InferenceSession.create('data:model/octet-stream;base64,{b64}');
        status.className = 'status success';
        status.textContent = '{best_model_name} loaded!';
        document.getElementById('badge').textContent = '{best_model_name} (Best Model)';
        
        // --- Metrics for the BEST MODEL ---
        document.getElementById('metricR2').textContent = '{best_model_metrics['r2']:.4f}';
        document.getElementById('metricRMSE').textContent = '${best_model_metrics['rmse']:,.0f}';
        document.getElementById('metricMAE').textContent = '${best_model_metrics['mae']:,.0f}';
        document.getElementById('metricModel').textContent = '{best_model_name}';
        // ----------------------------------

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
        
        // The input name 'input' matches what we defined in initial_types
        const input = new ort.Tensor('float32', new Float32Array(values), [1, 10]);
        const output = await session.run({{input: input}});
        
        // The default output name from onnxmltools is 'variable'
        const price = output.variable.data[0];

        document.getElementById('predictedPrice').textContent = '$' + Math.round(price).toLocaleString();
        document.getElementById('modelUsed').textContent = 'Using: {best_model_name} Model';
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

# --- 3. Inject new HTML and JavaScript into the file ---

# First, remove any old comparison table if it exists
html = re.sub(r'<div class="comparison-table">.*?</div>', '', html, flags=re.DOTALL)

# Inject the new comparison table AFTER the metrics-grid div
html = re.sub(
    r'(<div class="metrics-grid" id="metricsGrid">.*?</div>)',
    r'\1' + f'\n{comparison_table_html}\n',
    html,
    flags=re.DOTALL
)

# Second, replace the old script tag
html = re.sub(r'<script>.*</script>', new_script, html, flags=re.DOTALL)

# Write the updated HTML back to the file
Path("index.html").write_text(html, encoding="utf-8")

print("\nYOUR PROJECT IS NOW 100% WORKING")
print(f"Metrics for all models have been added to index.html.")
print(f"The best model ({best_model_name}) was embedded for prediction.")