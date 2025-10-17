"""
HealthTech Innovations — Patient Length-of-Stay (LOS) Prediction Lab
TensorFlow/Keras end‑to‑end: EDA → preprocessing → 3 optimizers → evaluation

How to use:
1) Ensure TensorFlow 2.x is installed:  pip install -U tensorflow scikit-learn joblib matplotlib
2) Put `patient_los.csv` in one of these locations:
   - /Users/karlkurzius/Downloads/patient_los.csv   (your local path)
   - ./patient_los.csv (same folder as this script)
3) Run this file as a script or copy cells into a notebook.
"""

# ============================
# Step 0: Libraries & Setup
# ============================
import os
import math
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, callbacks

# Reproducibility
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

print("TensorFlow:", tf.__version__)

# Where to look for the data
CANDIDATE_PATHS = [
    "/Users/karlkurzius/Downloads/patient_los.csv",  # local path mentioned in the prompt
    "./patient_los.csv",                               # current working directory
]

DATA_PATH = next((p for p in CANDIDATE_PATHS if os.path.exists(p)), None)
if DATA_PATH is None:
    raise FileNotFoundError("patient_los.csv not found. Place it in ./ or /Users/karlkurzius/Downloads/")

# ============================
# Step 1: Load & Explore Data
# ============================

df = pd.read_csv(DATA_PATH)
print(f"Rows: {len(df):,}  |  Cols: {df.shape[1]}")
print("Columns:", list(df.columns))

# Try to determine the target column automatically, but allow override
TARGET_COL_NAME = None  # set to e.g. "length_of_stay" if you know it exactly

if TARGET_COL_NAME is None:
    candidates = [
        "length_of_stay", "LOS", "los", "lengthofstay", "los_days",
        "length_of_stay_days", "stay_length", "stay_length_days"
    ]
    lower_map = {c.lower(): c for c in df.columns}
    target_col = next((lower_map[c.lower()] for c in candidates if c.lower() in lower_map), None)
    if target_col is None:
        # Fallback guess: any numeric column containing "los" or "stay"
        for c in df.columns:
            if df[c].dtype.kind in "if" and ("los" in c.lower() or "stay" in c.lower()):
                target_col = c
                break
    if target_col is None:
        # Last resort: pick the first numeric column
        numeric_candidates = [c for c in df.columns if df[c].dtype.kind in "if"]
        target_col = numeric_candidates[0] if numeric_candidates else None
else:
    if TARGET_COL_NAME not in df.columns:
        raise ValueError(f"Specified TARGET_COL_NAME='{TARGET_COL_NAME}' not found in columns")
    target_col = TARGET_COL_NAME

if target_col is None:
    raise ValueError("Could not infer LOS target column. Please set TARGET_COL_NAME explicitly.")

print("Target column:", target_col)

# Simple EDA: target distribution & peek at common categorical fields
plt.figure()
df[target_col].dropna().plot(kind="hist", bins=30, edgecolor="black", title=f"Distribution of {target_col}")
plt.xlabel(target_col); plt.ylabel("Count"); plt.tight_layout(); plt.show()

for col in ["admission_type", "insurance_type", "diagnosis_code"]:
    if col in df.columns and df[col].notna().any():
        print(f"\nTop 10 values for {col}:")
        print(df[col].value_counts().head(10))

# ============================
# Step 2: Preprocess for Training
# ============================

df = df[df[target_col].notna()].copy()
X = df.drop(columns=[target_col])
y = df[target_col].astype(float)

numeric_features = [c for c in X.columns if X[c].dtype.kind in "if"]
categorical_features = [c for c in X.columns if X[c].dtype.kind not in "if"]
print(f"Numeric features: {len(numeric_features)}  |  Categorical features: {len(categorical_features)}")

num_imputer = SimpleImputer(strategy="median")
cat_imputer = SimpleImputer(strategy="most_frequent")

X_num = pd.DataFrame(num_imputer.fit_transform(X[numeric_features]) if numeric_features else np.empty((len(X),0)),
                     columns=numeric_features, index=X.index)
X_cat = pd.DataFrame(cat_imputer.fit_transform(X[categorical_features]) if categorical_features else np.empty((len(X),0)),
                     columns=categorical_features, index=X.index)
X_clean = pd.concat([X_num, X_cat], axis=1)

preprocess = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numeric_features),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse=False), categorical_features),
    ],
    remainder="drop",
)

X_train, X_test, y_train, y_test = train_test_split(X_clean, y, test_size=0.2, random_state=SEED)
X_train_proc = preprocess.fit_transform(X_train)
X_test_proc = preprocess.transform(X_test)

feature_names = []
feature_names.extend(numeric_features)
if len(categorical_features):
    ohe = preprocess.named_transformers_["cat"]
    if hasattr(ohe, "get_feature_names_out"):
        feature_names.extend(ohe.get_feature_names_out(categorical_features).tolist())
    else:
        feature_names.extend([f"cat_{i}" for i in range(X_train_proc.shape[1] - len(numeric_features))])

print("Final feature dimension:", X_train_proc.shape[1])

# ============================
# Step 3: Build a Keras Model (Regression)
# ============================

def build_model(input_dim: int, hidden=(128,64,32), dropout_rate=0.1):
    model = keras.Sequential([
        layers.Input(shape=(input_dim,)),
        layers.Dense(hidden[0], activation="relu"),
        layers.Dropout(dropout_rate),
        layers.Dense(hidden[1], activation="relu"),
        layers.Dropout(dropout_rate),
        layers.Dense(hidden[2], activation="relu"),
        layers.Dropout(dropout_rate),
        layers.Dense(1, activation="linear"),
    ])
    return model

# ============================
# Step 4: Configure Training
# ============================

EPOCHS = 100
BATCH_SIZE = 128
early_stop = callbacks.EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)

optimizers = {
    "SGD": keras.optimizers.SGD(learning_rate=0.01, momentum=0.9, nesterov=True),
    "Adam": keras.optimizers.Adam(learning_rate=0.001),
    "RMSprop": keras.optimizers.RMSprop(learning_rate=0.001),
}

# ============================
# Step 5: Train Three Models & Analyze Learning
# ============================

histories = {}
models = {}
input_dim = X_train_proc.shape[1]

for opt_name, opt in optimizers.items():
    print(f"\nTraining with optimizer: {opt_name}")
    model = build_model(input_dim)
    model.compile(optimizer=opt, loss="mse", metrics=[keras.metrics.MAE, keras.metrics.RootMeanSquaredError()])
    hist = model.fit(
        X_train_proc, y_train.values,
        validation_split=0.2,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        verbose=1,
        callbacks=[early_stop]
    )
    histories[opt_name] = hist
    models[opt_name] = model

    # Learning curve (loss)
    plt.figure()
    plt.plot(hist.history["loss"], label="train_loss")
    plt.plot(hist.history["val_loss"], label="val_loss")
    plt.title(f"Learning Curve — {opt_name}")
    plt.xlabel("Epoch"); plt.ylabel("MSE Loss"); plt.legend(); plt.tight_layout(); plt.show()

# ============================
# Step 6: Evaluate on Test Data
# ============================

rows = []
for name, model in models.items():
    preds = model.predict(X_test_proc, verbose=0).ravel()
    mae = mean_absolute_error(y_test, preds)
    rmse = math.sqrt(mean_squared_error(y_test, preds))
    r2 = r2_score(y_test, preds)
    rows.append({"model": name, "MAE": mae, "RMSE": rmse, "R2": r2})

metrics_df = pd.DataFrame(rows).sort_values(by=["RMSE", "MAE"], ascending=[True, True])
print("\nTest metrics (lower MAE/RMSE is better, higher R2 is better):")
print(metrics_df)

best_name = metrics_df.iloc[0]["model"]
best_model = models[best_name]
print(f"\nBest model by RMSE/MAE: {best_name}")

# ============================
# Step 7: Final Evaluation, Save Artifacts, Stakeholder Notes
# ============================

# Save best model and preprocessing
os.makedirs("artifacts", exist_ok=True)
best_model_path = os.path.join("artifacts", f"best_los_model_{best_name}.keras")
preprocess_path = os.path.join("artifacts", "preprocess.joblib")
metrics_path = os.path.join("artifacts", "los_model_metrics.csv")

best_model.save(best_model_path)
joblib.dump({
    "preprocess": preprocess,
    "num_imputer": num_imputer,
    "cat_imputer": cat_imputer,
    "numeric_features": numeric_features,
    "categorical_features": categorical_features,
    "feature_names": feature_names,
    "target_col": target_col,
}, preprocess_path)
metrics_df.to_csv(metrics_path, index=False)

print(f"\nSaved model → {best_model_path}")
print(f"Saved preprocess pipeline → {preprocess_path}")
print(f"Saved metrics → {metrics_path}")

# Small prediction demo for 5 random rows
rng = np.random.default_rng(SEED)
idx = rng.choice(len(X_test), size=min(5, len(X_test)), replace=False)
X_demo = X_test.iloc[idx].copy()
y_true_demo = y_test.iloc[idx].copy()

# Recreate preprocessing to ensure consistency
X_demo_num = pd.DataFrame(num_imputer.transform(X_demo[numeric_features]) if numeric_features else np.empty((len(X_demo),0)),
                          columns=numeric_features, index=X_demo.index)
X_demo_cat = pd.DataFrame(cat_imputer.transform(X_demo[categorical_features]) if categorical_features else np.empty((len(X_demo),0)),
                          columns=categorical_features, index=X_demo.index)
X_demo_clean = pd.concat([X_demo_num, X_demo_cat], axis=1)
X_demo_proc = preprocess.transform(X_demo_clean)

preds_demo = best_model.predict(X_demo_proc, verbose=0).ravel()
preview = X_demo.copy()
preview["true_LOS"] = y_true_demo.values
preview["pred_LOS"] = preds_demo
preview["abs_error"] = (preview["true_LOS"] - preview["pred_LOS"]).abs()

print("\nSample predictions (5 rows):")
print(preview)
