# cn_ai/scripts/data_preprocessing.py

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import joblib
import os

# -----------------------------
# 1. Set project paths (path-independent)
# -----------------------------
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

raw_data_dir = os.path.join(project_root, "data", "csvs")
preprocessed_dir = os.path.join(project_root, "data", "preprocessed")
model_dir = os.path.join(project_root, "models")

os.makedirs(preprocessed_dir, exist_ok=True)
os.makedirs(model_dir, exist_ok=True)

# -----------------------------
# 2. List CSV files
# -----------------------------
csv_files = [f for f in os.listdir(raw_data_dir) if f.endswith(".csv")]
if not csv_files:
    raise FileNotFoundError(f"No CSV files found in {raw_data_dir}")

# -----------------------------
# 3. Process each CSV
# -----------------------------
for file in csv_files:
    print(f"Processing {file} ...")
    df = pd.read_csv(os.path.join(raw_data_dir, file))
    
    # Strip whitespace from column names
    df.columns = df.columns.str.strip()
    
    # -----------------------------
    # 4. Detect label column
    # -----------------------------
    label_cols = [col for col in df.columns if 'label' in col.lower()]
    if not label_cols:
        raise ValueError(f"No label column found in {file}")
    label_col = label_cols[0]
    print(f"Detected label column: {label_col}")
    
    # Preserve multi-class labels
    y = df[label_col].astype(str)
    
    # -----------------------------
    # 5. Feature selection
    # -----------------------------
    selected_features = [
        'Destination Port', 'Flow Duration', 'Total Fwd Packets', 'Total Backward Packets',
        'Total Length of Fwd Packets', 'Total Length of Bwd Packets',
        'Fwd Packet Length Max', 'Fwd Packet Length Min', 'Fwd Packet Length Mean', 'Fwd Packet Length Std',
        'Bwd Packet Length Max', 'Bwd Packet Length Min', 'Bwd Packet Length Mean', 'Bwd Packet Length Std',
        'Flow Bytes/s', 'Flow Packets/s', 'Flow IAT Mean', 'Flow IAT Std',
        'Flow IAT Max', 'Flow IAT Min', 'Fwd IAT Total', 'Fwd IAT Mean', 'Fwd IAT Std',
        'Fwd IAT Max', 'Fwd IAT Min', 'Bwd IAT Total', 'Bwd IAT Mean', 'Bwd IAT Std',
        'Bwd IAT Max', 'Bwd IAT Min', 'Fwd PSH Flags', 'Bwd PSH Flags', 'Fwd URG Flags',
        'Bwd URG Flags', 'Fwd Header Length', 'Bwd Header Length'
    ]
    
    # Keep only existing features (avoid missing columns)
    selected_features = [f for f in selected_features if f in df.columns]
    X = df[selected_features].copy()
    
    # -----------------------------
    # 6. Handle NaN / Inf / extreme values
    # -----------------------------
    X.replace([np.inf, -np.inf], 0, inplace=True)
    X.fillna(0, inplace=True)
    X = X.clip(-1e6, 1e6)  # optional: clip extreme values
    
    # -----------------------------
    # 7. Feature scaling
    # -----------------------------
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Save scaler (overwrite each time)
    joblib.dump(scaler, os.path.join(model_dir, "scaler.pkl"))
    
    # -----------------------------
    # 8. PCA for correlated features
    # -----------------------------
    pca = PCA(n_components=min(15, X_scaled.shape[1]))
    X_pca = pca.fit_transform(X_scaled)
    joblib.dump(pca, os.path.join(model_dir, "pca_transformer.pkl"))
    
    # -----------------------------
    # 9. Create preprocessed DataFrame
    # -----------------------------
    pca_columns = [f"PCA_{i+1}" for i in range(X_pca.shape[1])]
    df_pca = pd.DataFrame(X_pca, columns=pca_columns)
    df_pca[label_col] = y  # preserve multi-class labels
    
    # -----------------------------
    # 10. Save preprocessed CSV
    # -----------------------------
    output_file = os.path.join(preprocessed_dir, file)
    df_pca.to_csv(output_file, index=False)
    print(f"Saved preprocessed CSV to {output_file}\n")

print("All CSVs processed successfully!")
