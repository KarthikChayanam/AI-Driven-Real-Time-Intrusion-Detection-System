# scripts/test_pipeline.py

import os
import pandas as pd
import joblib
from sklearn.metrics import classification_report

# ---------------------------
# Paths
# ---------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROCESSED_FILE = os.path.join(BASE_DIR, "../data/processed/cicids2017_preprocessed.csv")
MODELS_DIR = os.path.join(BASE_DIR, "../models")

# ---------------------------
# Load model & scaler (and PCA if exists)
# ---------------------------
model = joblib.load(os.path.join(MODELS_DIR, "rf_model.pkl"))
scaler = joblib.load(os.path.join(MODELS_DIR, "scaler.pkl"))

pca_file = os.path.join(MODELS_DIR, "pca_transformer.pkl")
USE_PCA = os.path.exists(pca_file)
if USE_PCA:
    pca = joblib.load(pca_file)

# ---------------------------
# Load dataset for testing
# ---------------------------
df = pd.read_csv(PROCESSED_FILE)
X = df.drop(columns=["Label"])
y = df["Label"]

# ---------------------------
# Scale features
# ---------------------------
X_scaled = scaler.transform(X)

# ---------------------------
# Apply PCA if used
# ---------------------------
if USE_PCA:
    X_final = pca.transform(X_scaled)
else:
    X_final = X_scaled

# ---------------------------
# Predict
# ---------------------------
y_pred = model.predict(X_final)

# ---------------------------
# Classification report
# ---------------------------
print("🔹 Classification report on full dataset:")
print(classification_report(y, y_pred))

# ---------------------------
# Simulate real-time alert logging
# ---------------------------
print("\n🔹 Simulating real-time alerts...")
for idx, pred_label in enumerate(y_pred[:100000]):  # limit to first 1000 for demo
    true_label = y.iloc[idx]
    if pred_label != 0:  # assuming 0 = BENIGN
        print(f"⚠ Alert! Attack detected: Predicted={pred_label}, Actual={true_label}")
