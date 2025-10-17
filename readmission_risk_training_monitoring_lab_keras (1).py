"""
HealthTech Analytics — Readmission Risk Training Monitoring Lab (Keras)

This script mirrors the lab steps and adds a robust training monitoring suite:
- Baseline model (no callbacks) → metrics + curves
- Improved model with callbacks: EarlyStopping, ModelCheckpoint, TensorBoard, ReduceLROnPlateau,
  TerminateOnNaN, gradient clipping, Dropout
- Visual diagnostics: loss/accuracy curves, ROC‑AUC, PR‑AUC, confusion matrix, calibration
- Reproducible preprocessing (ColumnTransformer with imputation + scaling + one‑hot)

How to run locally
------------------
1) Install deps:
   pip install -U tensorflow scikit-learn pandas numpy matplotlib seaborn joblib
2) Ensure data file exists at one of:
   - /Users/karlkurzius/Downloads/readmission_data.csv  (as per prompt)
   - ./readmission_data.csv  (same folder as this script)
3) (Optional) Launch TensorBoard in a separate terminal:
   tensorboard --logdir tb_logs
4) Run this script (python) or copy cells into your notebook.
"""

# ============================
# Step 0: Imports & Setup
# ============================
import os
import math
import time
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score, roc_auc_score, average_precision_score,
    confusion_matrix, classification_report
)
from sklearn.calibration import calibration_curve

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, callbacks

SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

ARTIFACTS_DIR = Path("artifacts"); ARTIFACTS_DIR.mkdir(exist_ok=True)
TB_DIR = Path("tb_logs"); TB_DIR.mkdir(exist_ok=True)

# ============================
# Step 1: Load & Understand the Dataset
# ============================
CANDIDATE_PATHS = [
    "/Users/karlkurzius/Downloads/readmission_data.csv",
    "./readmission_data.csv",
]
DATA_PATH = next((p for p in CANDIDATE_PATHS if os.path.exists(p)), None)
if DATA_PATH is None:
    raise FileNotFoundError("readmission_data.csv not found in ./ or /Users/karlkurzius/Downloads/")

# Load
raw = pd.read_csv(DATA_PATH)
print(f"Rows: {len(raw):,} | Cols: {raw.shape[1]}")
print("Columns:", list(raw.columns)[:25], ("..." if raw.shape[1] > 25 else ""))

# Infer target name (binary classification). Adjust if your dataset differs.
TARGET_NAME = None  # set manually if known, e.g., "readmitted"
if TARGET_NAME is None:
    # Guess common target names
    guesses = ["readmitted","readmission","readmit","target","label","y"]
    lower_map = {c.lower(): c for c in raw.columns}
    TARGET_NAME = next((lower_map[g] for g in guesses if g in lower_map), None)
if TARGET_NAME is None:
    raise ValueError("Could not infer TARGET_NAME. Set TARGET_NAME to your label column.")

# Basic checks
y_raw = raw[TARGET_NAME].copy()
if y_raw.dtype.kind in "biu":
    y = y_raw.astype(int)
else:
    # If strings like 'Yes'/'No'
    y = y_raw.astype(str).str.lower().isin(["1","true","yes","y"]).astype(int)

X = raw.drop(columns=[TARGET_NAME])

# Feature typing
num_cols = [c for c in X.columns if X[c].dtype.kind in "if"]
cat_cols = [c for c in X.columns if X[c].dtype.kind not in "if"]
print(f"Numeric features: {len(num_cols)} | Categorical features: {len(cat_cols)}")

# Class balance
pos_rate = y.mean()
print(f"Positive rate (readmitted=1): {pos_rate:.3f}")

# Distributions (quick glance)
plt.figure(); y.value_counts().plot(kind="bar"); plt.title("Class Balance"); plt.xticks(rotation=0); plt.tight_layout(); plt.show()

# ============================
# Step 2: Preprocess & Baseline Model (no callbacks)
# ============================
num_imputer = SimpleImputer(strategy="median")
cat_imputer = SimpleImputer(strategy="most_frequent")

X_num = pd.DataFrame(num_imputer.fit_transform(X[num_cols]) if num_cols else np.empty((len(X),0)),
                     columns=num_cols, index=X.index)
X_cat = pd.DataFrame(cat_imputer.fit_transform(X[cat_cols]) if cat_cols else np.empty((len(X),0)),
                     columns=cat_cols, index=X.index)
X_clean = pd.concat([X_num, X_cat], axis=1)

preprocess = ColumnTransformer([
    ("num", StandardScaler(), num_cols),
    ("cat", OneHotEncoder(handle_unknown="ignore", sparse=False), cat_cols),
], remainder="drop")

# Train/Val/Test split (stratified)
X_trainval, X_test, y_trainval, y_test = train_test_split(
    X_clean, y, test_size=0.2, random_state=SEED, stratify=y
)
X_train, X_val, y_train, y_val = train_test_split(
    X_trainval, y_trainval, test_size=0.2, random_state=SEED, stratify=y_trainval
)

# Fit transform
X_train_p = preprocess.fit_transform(X_train)
X_val_p   = preprocess.transform(X_val)
X_test_p  = preprocess.transform(X_test)

input_dim = X_train_p.shape[1]

# Baseline architecture
def build_baseline(input_dim: int):
    model = keras.Sequential([
        layers.Input(shape=(input_dim,)),
        layers.Dense(64, activation="relu"),
        layers.Dense(1, activation="sigmoid"),
    ])
    return model

baseline = build_baseline(input_dim)
baseline.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-3),
    loss="binary_crossentropy",
    metrics=[keras.metrics.BinaryAccuracy(name="accuracy")]
)

hist_base = baseline.fit(
    X_train_p, y_train,
    validation_data=(X_val_p, y_val),
    epochs=50,
    batch_size=256,
    verbose=1,
)

# ============================
# Step 3: Visualize Training Curves (Baseline)
# ============================

def plot_curves(history, title_prefix=""):
    h = history.history
    # Loss
    plt.figure()
    plt.plot(h["loss"], label="train")
    plt.plot(h.get("val_loss", []), label="val")
    plt.title(f"{title_prefix}Loss")
    plt.xlabel("Epoch"); plt.ylabel("Binary Crossentropy"); plt.legend(); plt.tight_layout(); plt.show()
    # Accuracy
    if "accuracy" in h:
        plt.figure()
        plt.plot(h["accuracy"], label="train")
        if "val_accuracy" in h:
            plt.plot(h["val_accuracy"], label="val")
        plt.title(f"{title_prefix}Accuracy")
        plt.xlabel("Epoch"); plt.ylabel("Accuracy"); plt.legend(); plt.tight_layout(); plt.show()

plot_curves(hist_base, title_prefix="Baseline – ")

# Helper: evaluation report
from sklearn.metrics import precision_recall_curve, roc_curve

def evaluate_model(model, Xp, y_true, name="model", threshold=0.5):
    p = model.predict(Xp, verbose=0).ravel()
    auc = roc_auc_score(y_true, p)
    ap  = average_precision_score(y_true, p)
    y_hat = (p >= threshold).astype(int)
    acc = accuracy_score(y_true, y_hat)
    cm = confusion_matrix(y_true, y_hat)
    print(f"\n[{name}]  ACC={acc:.3f}  ROC-AUC={auc:.3f}  PR-AUC={ap:.3f}")
    print("Confusion Matrix:\n", cm)
    print(classification_report(y_true, y_hat, digits=3))

    # ROC curve
    fpr, tpr, _ = roc_curve(y_true, p)
    plt.figure(); plt.plot(fpr, tpr); plt.plot([0,1],[0,1],'--'); plt.title(f"ROC – {name}"); plt.xlabel("FPR"); plt.ylabel("TPR"); plt.tight_layout(); plt.show()
    # PR curve
    prec, rec, _ = precision_recall_curve(y_true, p)
    plt.figure(); plt.plot(rec, prec); plt.title(f"PR – {name}"); plt.xlabel("Recall"); plt.ylabel("Precision"); plt.tight_layout(); plt.show()
    # Calibration
    prob_true, prob_pred = calibration_curve(y_true, p, n_bins=10, strategy="quantile")
    plt.figure(); plt.plot(prob_pred, prob_true, marker='o'); plt.plot([0,1],[0,1],'--'); plt.title(f"Calibration – {name}"); plt.xlabel("Predicted prob"); plt.ylabel("True rate"); plt.tight_layout(); plt.show()

print("\n=== Baseline Evaluation ===")
evaluate_model(baseline, X_val_p, y_val, name="Baseline (Val)")
evaluate_model(baseline, X_test_p, y_test, name="Baseline (Test)")

# ============================
# Step 4: Callbacks & Gradient Issues (Improved Model)
# ============================

def build_improved(input_dim: int, dropout=0.3):
    return keras.Sequential([
        layers.Input(shape=(input_dim,)),
        layers.Dense(256, activation="relu"),
        layers.Dropout(dropout),
        layers.Dense(128, activation="relu"),
        layers.Dropout(dropout),
        layers.Dense(64, activation="relu"),
        layers.Dropout(dropout/2),
        layers.Dense(1, activation="sigmoid"),
    ])

# Optimizer w/ gradient clipping
opt = keras.optimizers.Adam(learning_rate=1e-3, clipnorm=1.0)  # or clipvalue=1.0

improved = build_improved(input_dim)
improved.compile(
    optimizer=opt,
    loss="binary_crossentropy",
    metrics=[keras.metrics.BinaryAccuracy(name="accuracy")]
)

# EarlyStopping + ModelCheckpoint + TensorBoard + ReduceLROnPlateau + TerminateOnNaN
run_id = time.strftime("%Y%m%d_%H%M%S")
ckpt_path = ARTIFACTS_DIR / f"best_readmit_{run_id}.keras"
cb = [
    callbacks.EarlyStopping(monitor="val_loss", patience=10, min_delta=1e-4, restore_best_weights=True),
    callbacks.ModelCheckpoint(filepath=str(ckpt_path), monitor="val_loss", save_best_only=True),
    callbacks.TensorBoard(log_dir=str(TB_DIR / f"run_{run_id}"), histogram_freq=1, write_graph=True),
    callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5, min_lr=1e-6, verbose=1),
    callbacks.TerminateOnNaN(),
]

# Optional: class weights for imbalance
pos_weight = (len(y_train) - y_train.sum()) / max(y_train.sum(), 1)
class_weight = {0:1.0, 1:float(pos_weight)} if pos_weight > 1.0 else None
print("Class weight:", class_weight)

hist_imp = improved.fit(
    X_train_p, y_train,
    validation_data=(X_val_p, y_val),
    epochs=100,
    batch_size=256,
    callbacks=cb,
    verbose=1,
    class_weight=class_weight,
)

plot_curves(hist_imp, title_prefix="Improved – ")

print("\n=== Improved Evaluation ===")
evaluate_model(improved, X_val_p, y_val, name="Improved (Val)")
evaluate_model(improved, X_test_p, y_test, name="Improved (Test)")

# Save preprocessing for reuse
joblib.dump({
    "preprocess": preprocess,
    "num_imputer": num_imputer,
    "cat_imputer": cat_imputer,
    "numeric_features": num_cols,
    "categorical_features": cat_cols,
    "target": TARGET_NAME
}, ARTIFACTS_DIR / "preprocess.joblib")
print("Saved:", ckpt_path, "and preprocess.joblib")

# ============================
# Step 5: Compare results (with vs without EarlyStopping)
# ============================
print("\n=== Comparison (Val set) ===")
# Note: Baseline had no early stopping; Improved has it.
# You can compare AUC/PR‑AUC directly above from evaluate_model outputs.

# ============================
# Step 6: Reflection Prompts (fill these in markdown cells of your notebook)
# -------------------------------------------------------------------------
# 1) How did early stopping affect the training process and final model performance?
# 2) What patterns did you observe in the training and validation curves?
# 3) In a healthcare context, why is it particularly important to prevent overfitting?
# 4) How would you explain the benefits of your monitoring approach to non‑technical staff?
