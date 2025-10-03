# scripts/data_preprocessing.py

import os
import sys
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA

# ---------------------------
# Step 0: Paths relative to script
# ---------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # directory of this script
RAW_DATA_DIR = os.path.join(BASE_DIR, "../data/csvs")
PROCESSED_DATA_DIR = os.path.join(BASE_DIR, "../data/processed")
PROCESSED_FILE = os.path.join(PROCESSED_DATA_DIR, "cicids2017_preprocessed.csv")

# ---------------------------
# Step 1: Verify CSV folder
# ---------------------------
if not os.path.exists(RAW_DATA_DIR):
    print(f"[ERROR] CSV folder not found: {RAW_DATA_DIR}")
    print("Please create the folder and add your CIC-IDS-2017 CSV files.")
    sys.exit(1)

csv_files = [f for f in os.listdir(RAW_DATA_DIR) if f.endswith(".csv")]
if not csv_files:
    print(f"[ERROR] No CSV files found in {RAW_DATA_DIR}")
    sys.exit(1)

print("🔹 Loading CSV files...")
dataframes = []
for file in csv_files:
    path = os.path.join(RAW_DATA_DIR, file)
    print(f"  - Reading {file}")
    df = pd.read_csv(path)
    dataframes.append(df)

df = pd.concat(dataframes, ignore_index=True)
print(f"✅ Combined dataset shape: {df.shape}")

# ---------------------------
# Step 2: Clean labels
# ---------------------------
print("🔹 Cleaning labels...")
if " Label" in df.columns:
    df.rename(columns={" Label": "Label"}, inplace=True)

df["Label"] = df["Label"].astype(str).str.strip()  # remove extra spaces

# ---------------------------
# Step 3: Encode labels
# ---------------------------
print("🔹 Encoding labels...")
label_encoder = LabelEncoder()
df["Label"] = label_encoder.fit_transform(df["Label"])
print(f"✅ Classes: {list(label_encoder.classes_)}")

# ---------------------------
# Step 4: Split features & labels
# ---------------------------
X = df.drop(columns=["Label"])
y = df["Label"]

# ---------------------------
# Step 5: Clean infinite / NaN / extreme values
# ---------------------------
print("🔹 Cleaning infinite / NaN values...")
X.replace([np.inf, -np.inf], np.nan, inplace=True)
X.fillna(X.median(), inplace=True)

# Optional: clip extreme values
X = X.clip(lower=-1e6, upper=1e6)

# ---------------------------
# Step 6: Scale features
# ---------------------------
print("🔹 Scaling features...")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ---------------------------
# Step 7: PCA (optional)
# ---------------------------
USE_PCA = True
PCA_COMPONENTS = 30

if USE_PCA:
    print(f"🔹 Applying PCA (n_components={PCA_COMPONENTS})...")
    pca = PCA(n_components=PCA_COMPONENTS, random_state=42)
    X_reduced = pca.fit_transform(X_scaled)
    X_final = pd.DataFrame(X_reduced, columns=[f"PC{i+1}" for i in range(PCA_COMPONENTS)])
    print(f"✅ PCA reduced shape: {X_final.shape}")
else:
    X_final = pd.DataFrame(X_scaled, columns=X.columns)

# ---------------------------
# Step 8: Save processed dataset
# ---------------------------
os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
processed_df = pd.concat([X_final, y.reset_index(drop=True)], axis=1)
processed_df.to_csv(PROCESSED_FILE, index=False)
print(f"✅ Preprocessed dataset saved at: {PROCESSED_FILE}")
