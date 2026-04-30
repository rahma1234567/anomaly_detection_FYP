from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parent.parent
SCALER_DIR    = ROOT / "saved_scalers"
MODEL_DIR     = ROOT / "models" / "saved_models"

LOC_FREQ_PATH = SCALER_DIR / "location_frequency.pkl"
AE_SCALER_PATH = SCALER_DIR / "autoen_minmax_scaler.pkl"
IF_MODEL_PATH = MODEL_DIR / "isolation_forest.pkl"
AE_MODEL_PATH = MODEL_DIR / "autoencoder.pth"
AE_META_PATH  = MODEL_DIR / "autoencoder_metadata.pkl"

REQUIRED_COLUMNS = [
    "TransactionID", "AccountID", "TransactionAmount", "TransactionDate",
    "TransactionType", "Location", "DeviceID", "IP Address", "MerchantID",
    "Channel", "CustomerAge", "CustomerOccupation", "TransactionDuration",
    "LoginAttempts", "AccountBalance", "PreviousTransactionDate",]

CONTINUOUS_COLS = [
    "TransactionAmount", "CustomerAge", "TransactionDuration",
    "AccountBalance", "LoginAttempts", "days_since_last_transaction",
    "amount_balance_ratio", "quick_transactions", "Location",]
CYCLIC_COLS = ["hour_sin", "hour_cos", "week_sin", "week_cos", "month_sin", "month_cos"]

# Autoencoder definition
class Autoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim,  hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim), nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),  nn.Sigmoid(),)

    def forward(self, x):
        return self.decoder(self.encoder(x))

def validate_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in REQUIRED_COLUMNS if c not in df.columns]
def _preprocess(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    df  = df.copy()
    raw = df.copy()

    # Parse dates and sort (like training)
    df["TransactionDate"] = pd.to_datetime(df["TransactionDate"], dayfirst=True, errors="coerce")
    df = df.sort_values(["AccountID", "TransactionDate"]).reset_index(drop=True)
    raw = raw.loc[df.index].reset_index(drop=True)  # keep raw aligned

    # Per-account days-since-last-transaction
    df["days_since_last_transaction"] = (
        df.groupby("AccountID")["TransactionDate"]
          .diff()
          .dt.total_seconds() / 86400.0).fillna(0)

    # Cyclical time encoding
    hour  = df["TransactionDate"].dt.hour
    week  = df["TransactionDate"].dt.dayofweek
    month = df["TransactionDate"].dt.month
    df["hour_sin"]  = np.sin(2 * np.pi * hour  / 24)
    df["hour_cos"]  = np.cos(2 * np.pi * hour  / 24)
    df["week_sin"]  = np.sin(2 * np.pi * week  / 7)
    df["week_cos"]  = np.cos(2 * np.pi * week  / 7)
    df["month_sin"] = np.sin(2 * np.pi * month / 12)
    df["month_cos"] = np.cos(2 * np.pi * month / 12)
    df["transaction_hour"] = hour 

    # Drop ID and date columns
    drop_cols = ["TransactionID", "AccountID", "DeviceID", "IP Address",
                 "MerchantID", "TransactionDate", "PreviousTransactionDate"]
    df.drop(columns=[c for c in drop_cols if c in df.columns], inplace=True)

    # Encode categoricals
    df["TransactionType"] = (df["TransactionType"] == "Debit").astype(int)
    df = pd.get_dummies(df, columns=["Channel", "CustomerOccupation"], drop_first=False)
    bool_cols = df.select_dtypes(include="bool").columns
    df[bool_cols] = df[bool_cols].astype(int)

    loc_freq = joblib.load(LOC_FREQ_PATH)
    df["Location"] = df["Location"].map(loc_freq).fillna(0)

    return df, raw

def _add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    safe_balance  = np.clip(df["AccountBalance"].to_numpy(),    1.0, None)
    safe_duration = np.clip(df["TransactionDuration"].to_numpy(), 1.0, None)
    df["amount_balance_ratio"] = df["TransactionAmount"] / safe_balance
    df["night_transaction"]    = (df["transaction_hour"] <= 5).astype(int)
    df["high_login_attempts"]  = (df["LoginAttempts"] >= 3).astype(int)
    df["quick_transactions"]   = df["TransactionAmount"] / safe_duration
    df.drop(columns=["transaction_hour"], inplace=True)
    return df

def _to_ae_space(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    scaler = joblib.load(AE_SCALER_PATH)
    df[CONTINUOUS_COLS] = scaler.transform(df[CONTINUOUS_COLS])
    df[CONTINUOUS_COLS] = df[CONTINUOUS_COLS].clip(0, 1)
    for col in CYCLIC_COLS:
        df[col] = (df[col] + 1) / 2
    return df

#run all the models
def _align_to_model(df: pd.DataFrame, expected_cols) -> pd.DataFrame:
    """Add any missing one-hot columns as 0s, then put in the right order."""
    for col in expected_cols:
        if col not in df.columns:
            df[col] = 0
    return df[list(expected_cols)]

def _run_isolation_forest(df: pd.DataFrame):
    model = joblib.load(IF_MODEL_PATH)
    aligned = _align_to_model(df, model.feature_names_in_)
    raw_scores  = model.score_samples(aligned)
    predictions = model.predict(aligned)
    threshold = float(-model.offset_)
    return -raw_scores, (predictions == -1).astype(int), threshold

def _run_autoencoder(df: pd.DataFrame):
    meta = joblib.load(AE_META_PATH)
    aligned = _align_to_model(df, meta["feature_columns"])

    model = Autoencoder(meta["input_dim"], meta["hidden_dim"], meta["latent_dim"])
    model.load_state_dict(torch.load(AE_MODEL_PATH, map_location="cpu", weights_only=True))
    model.eval()
    x = torch.tensor(aligned.values.astype("float32"), dtype=torch.float32)
    with torch.no_grad():
        recon = model(x)
    errors = ((x - recon) ** 2).mean(dim=1).numpy()
    threshold = float(meta["threshold"])
    is_anomaly = (errors > threshold).astype(int)
    return errors, is_anomaly, threshold

def run_pipeline(df: pd.DataFrame, model_name: str):
    processed, raw = _preprocess(df)
    engineered = _add_features(processed)

    if model_name == "Isolation Forest":
        scores, flags, threshold = _run_isolation_forest(engineered)
        score_col = "anomaly_score"
    else:
        scaled = _to_ae_space(engineered)
        scores, flags, threshold = _run_autoencoder(scaled)
        score_col = "reconstruction_error"

    raw[score_col]    = scores
    raw["is_anomaly"] = flags
    return raw, score_col, threshold