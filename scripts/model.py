# scripts/models.py

import os
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# ---------------------------
# Paths
# ---------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROCESSED_FILE = os.path.join(BASE_DIR, "../data/processed/cicids2017_preprocessed.csv")
MODELS_DIR = os.path.join(BASE_DIR, "../models")

os.makedirs(MODELS_DIR, exist_ok=True)

# ---------------------------
# Load preprocessed data
# ---------------------------
print("🔹 Loading preprocessed dataset...")
df = pd.read_csv(PROCESSED_FILE)
print(f"✅ Dataset shape: {df.shape}")

# ---------------------------
# Split features and labels
# ---------------------------
X = df.drop(columns=["Label"])
y = df["Label"]

# ---------------------------
# Train/test split
# ---------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"✅ Training samples: {X_train.shape[0]}, Testing samples: {X_test.shape[0]}")

# ---------------------------
# Scale features
# ---------------------------
print("🔹 Scaling features...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ---------------------------
# Optional: PCA
# ---------------------------
USE_PCA = False      # Set True if you want dimensionality reduction
PCA_COMPONENTS = 30

if USE_PCA:
    print(f"🔹 Applying PCA (n_components={PCA_COMPONENTS})...")
    pca = PCA(n_components=PCA_COMPONENTS, random_state=42)
    X_train_final = pca.fit_transform(X_train_scaled)
    X_test_final = pca.transform(X_test_scaled)
else:
    X_train_final = X_train_scaled
    X_test_final = X_test_scaled

# ---------------------------
# Train RandomForest
# ---------------------------
print("🔹 Training RandomForestClassifier...")
rf_model = RandomForestClassifier(
    n_estimators=200,
    max_depth=20,
    random_state=42,
    n_jobs=-1
)
rf_model.fit(X_train_final, y_train)
print("✅ Model trained successfully.")

# ---------------------------
# Evaluate
# ---------------------------
y_pred = rf_model.predict(X_test_final)
print("🔹 Classification report:")
print(classification_report(y_test, y_pred))

# ---------------------------
# Save model & scaler (and PCA if used)
# ---------------------------
model_file = os.path.join(MODELS_DIR, "rf_model.pkl")
joblib.dump(rf_model, model_file)
print(f"✅ Model saved at: {model_file}")

scaler_file = os.path.join(MODELS_DIR, "scaler.pkl")
joblib.dump(scaler, scaler_file)
print(f"✅ Scaler saved at: {scaler_file}")

if USE_PCA:
    pca_file = os.path.join(MODELS_DIR, "pca_transformer.pkl")
    joblib.dump(pca, pca_file)
    print(f"✅ PCA transformer saved at: {pca_file}")
