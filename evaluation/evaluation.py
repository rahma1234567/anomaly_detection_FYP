# Synthetic anomaly evaluation for Isolation Forest + Autoencoder.
import os
import sys
import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import (precision_score, recall_score, f1_score, roc_auc_score, 
                             average_precision_score, confusion_matrix,)

AE_MODEL_PATH    = "models/saved_models/autoencoder.pth"
AE_META_PATH     = "models/saved_models/autoencoder_metadata.pkl"
IF_MODEL_PATH    = "models/saved_models/isolation_forest.pkl"   
AE_SCALER_PATH    = "saved_scalers/autoen_minmax_scaler.pkl" 

X_TRAIN_RAW_CSV  = "feature_engineering/X_train_isolation.csv"
X_TEST_RAW_CSV   = "feature_engineering/X_test_isolation.csv"
X_TRAIN_AE_CSV   = "feature_engineering/X_train_autoen.csv"
X_TEST_AE_CSV    = "feature_engineering/X_test_autoen.csv"

CYCLIC_COLS      = ["hour_sin", "hour_cos", "week_sin", "week_cos", "month_sin", "month_cos"]
CONTINUOUS_COLS  = ["TransactionAmount", "CustomerAge", "TransactionDuration",
                     "AccountBalance", "LoginAttempts", "days_since_last_transaction",
                    "amount_balance_ratio", "quick_transactions", "Location",]

HIGH_LOGIN_THRESHOLD = 4   
N_PER_LEVEL          = 10  # 10 anomalies per (pattern, severity)
RNG = np.random.default_rng(42)

# Loading the model, scalars and data
iforest   = joblib.load(IF_MODEL_PATH)
ae_meta   = joblib.load(AE_META_PATH)
ae_scaler = joblib.load(AE_SCALER_PATH)

class Autoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim), nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, input_dim), nn.Sigmoid(),
        )
    def forward(self, x):
        return self.decoder(self.encoder(x))

ae = Autoencoder(ae_meta["input_dim"], ae_meta["hidden_dim"], ae_meta["latent_dim"])
ae.load_state_dict(torch.load(AE_MODEL_PATH))
ae.eval()
ae_threshold = ae_meta["threshold"]
ae_columns   = ae_meta["feature_columns"]

X_test_raw  = pd.read_csv(X_TEST_RAW_CSV)
X_train_raw = pd.read_csv(X_TRAIN_RAW_CSV)

#cyclic columns need to be in [0, 1] for the autoencoder
sample_min = X_train_raw[CYCLIC_COLS].min().min()
CYCLIC_NEEDS_SHIFT = sample_min < -0.001
print(f"Cyclic columns in iforest CSV are in {'[-1, 1] (will shift)' if CYCLIC_NEEDS_SHIFT else '[0, 1] (already shifted)'}")

# Scoring helpers
def score_iforest(df_raw):
    """Higher = more anomalous."""
    return -iforest.score_samples(df_raw)

def to_ae_space(df_raw):
    df_ae = df_raw.copy()
    df_ae[CONTINUOUS_COLS] = ae_scaler.transform(df_raw[CONTINUOUS_COLS])
    df_ae[CONTINUOUS_COLS] = df_ae[CONTINUOUS_COLS].clip(0, 1)
    if CYCLIC_NEEDS_SHIFT:
        for col in CYCLIC_COLS:
            df_ae[col] = (df_raw[col] + 1) / 2
    return df_ae[ae_columns]

def score_autoencoder(df_raw):
    df_ae = to_ae_space(df_raw)
    with torch.no_grad():
        x = torch.tensor(df_ae.values, dtype=torch.float32)
        return ((x - ae(x)) ** 2).mean(dim=1).numpy()

def recompute_engineered(df):
    """Re-derive engineered features after mutating raw columns."""
    df = df.copy()
    safe_balance  = np.clip(df["AccountBalance"].to_numpy(),    1.0, None)
    safe_duration = np.clip(df["TransactionDuration"].to_numpy(), 1.0, None)
    df["amount_balance_ratio"] = df["TransactionAmount"] / safe_balance
    df["quick_transactions"]   = df["TransactionAmount"] / safe_duration
    df["high_login_attempts"]  = (df["LoginAttempts"] >= HIGH_LOGIN_THRESHOLD).astype(int)
    if "night_transaction" in df.columns:
        if "transaction_hour" in df.columns:
            df["night_transaction"] = (df["transaction_hour"] <= 5).astype(int)
        #
    return df

# Synthetic anomaly patterns
def make_high_value(row, severity):
    row["TransactionAmount"] = row["TransactionAmount"] * [3.0, 7.0, 15.0][severity]
    return row

def make_account_drain(row, severity):
    row["TransactionAmount"] = row["AccountBalance"] * [0.60, 0.85, 0.98][severity]
    return row

def make_credential_stuffing(row, severity):
    row["LoginAttempts"] = [3, 4, 5][severity]
    return row

def make_rapid_micro(row, severity):
    row["TransactionDuration"] = [3, 2, 1][severity]
    return row

def make_combined(row, severity):
    row["TransactionAmount"] = row["TransactionAmount"] * [2.0, 4.0, 8.0][severity]
    row["LoginAttempts"]     = [3, 4, 5][severity]
    row["AccountBalance"]    = row["AccountBalance"]    * [0.50, 0.20, 0.05][severity]
    return row

PATTERNS = { "high_value":          make_high_value,
             "account_drain":       make_account_drain,
                "credential_stuffing": make_credential_stuffing,
             "rapid_micro":         make_rapid_micro,
             "combined":            make_combined,}

# Build labelled set
real = X_test_raw.reset_index(drop=True).copy()
real["label"]    = 0
real["pattern"]  = "real"
real["severity"] = -1

synthetic_rows = []
for pattern_name, mutator in PATTERNS.items():
    for severity in range(3):
        idx = RNG.choice(len(real), size=N_PER_LEVEL, replace=False)
        for i in idx:
            base = real.iloc[i].drop(["label", "pattern", "severity"]).copy()
            mutated = mutator(base.copy(), severity)
            mutated["label"]    = 1
            mutated["pattern"]  = pattern_name
            mutated["severity"] = severity
            synthetic_rows.append(mutated)

synthetic = pd.DataFrame(synthetic_rows).reset_index(drop=True)
synthetic = recompute_engineered(synthetic)

labelled = pd.concat([real, synthetic], ignore_index=True)
print(f"\nLabelled set: {len(labelled)} rows  ({(labelled['label']==0).sum()} real, {(labelled['label']==1).sum()} synthetic)")

# Score
feature_cols = [c for c in labelled.columns if c not in ("label", "pattern", "severity")]
features = labelled[feature_cols]

labelled["if_score"] = score_iforest(features)
labelled["ae_score"] = score_autoencoder(features)
labelled["if_pred"]  = (iforest.predict(features) == -1).astype(int)
labelled["ae_pred"]  = (labelled["ae_score"] > ae_threshold).astype(int)

y = labelled["label"].values

def metrics_block(name, y_pred, y_score):
    cm = confusion_matrix(y, y_pred)
    return {
        "model":     name,
        "accuracy":  (y == y_pred).mean(),
        "precision": precision_score(y, y_pred, zero_division=0),
        "recall":    recall_score(y, y_pred),
        "f1":        f1_score(y, y_pred),
        "roc_auc":   roc_auc_score(y, y_score),
        "pr_auc":    average_precision_score(y, y_score),
        "tn": cm[0,0], "fp": cm[0,1], "fn": cm[1,0], "tp": cm[1,1], }

metrics_df = pd.DataFrame([
    metrics_block("IsolationForest", labelled["if_pred"].values, labelled["if_score"].values),
    metrics_block("Autoencoder",     labelled["ae_pred"].values, labelled["ae_score"].values),
])

per_pattern = []
for pattern in PATTERNS:
    for severity in range(3):
        mask = (labelled["pattern"] == pattern) & (labelled["severity"] == severity)
        sub  = labelled[mask]
        per_pattern.append({
            "pattern":   pattern,
            "severity":  ["mild", "moderate", "severe"][severity],
            "n":         len(sub),
            "if_recall": sub["if_pred"].mean(),
            "ae_recall": sub["ae_pred"].mean(),
        })
per_pattern_df = pd.DataFrame(per_pattern)

# Threshold sweep
def sweep(name, scores):
    rows = []
    for q in [80, 85, 90, 92, 95, 97, 99]:
        thr  = np.percentile(scores, q)
        pred = (scores > thr).astype(int)
        rows.append({
            "model": name, "percentile": q, "threshold": float(thr),
            "precision": precision_score(y, pred, zero_division=0),
            "recall":    recall_score(y, pred),
            "f1":        f1_score(y, pred),
            "flagged":   int(pred.sum()),
        })
    return rows

sweep_df = pd.DataFrame( sweep("IsolationForest", labelled["if_score"].values) +
                        sweep("Autoencoder",     labelled["ae_score"].values))

# Save and report
os.makedirs("results", exist_ok=True)
labelled.to_csv("results/labelled_test.csv", index=False)
metrics_df.to_csv("results/evaluation_metrics.csv", index=False)
per_pattern_df.to_csv("results/per_pattern_recall.csv", index=False)
sweep_df.to_csv("results/threshold_sweep.csv", index=False)

print("\n")
print("METRICS")
print(metrics_df.round(4).to_string(index=False))

print("\n")
print("PER-PATTERN RECALL")
print(per_pattern_df.round(3).to_string(index=False))

print("\n" + "="*70)
print("THRESHOLD SWEEP")
print(sweep_df.round(4).to_string(index=False))

print("\nSaved: results/labelled_test.csv, evaluation_metrics.csv, per_pattern_recall.csv, threshold_sweep.csv")