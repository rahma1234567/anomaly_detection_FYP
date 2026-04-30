import shutil
from datetime import datetime
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.ensemble import IsolationForest
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parent.parent
SCALER_DIR = ROOT / "saved_scalers"
MODEL_DIR  = ROOT / "models" / "saved_models"
STAGING    = MODEL_DIR / "_staging"
BACKUPS    = MODEL_DIR / "_backups"
TRAIN_DATA_PATH = ROOT / "data" / "bank_transactions_data_2.csv"

# Training hyperparameters
N_EST = 200
CONTAMINATION = 0.05
HIDDEN, LATENT = 16, 8
EPOCHS, LR, BATCH = 200, 1e-3, 64

# Same Autoencoder definition
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
    def forward(self, x): return self.decoder(self.encoder(x))

def retrain(new_data: pd.DataFrame) -> dict:
    if not TRAIN_DATA_PATH.exists():
        raise FileNotFoundError(f"Original training CSV missing: {TRAIN_DATA_PATH}")

    STAGING.mkdir(parents=True, exist_ok=True)

    existing = pd.read_csv(TRAIN_DATA_PATH)
    combined = pd.concat([existing, new_data], ignore_index=True)
    combined = combined.drop_duplicates(subset="TransactionID", keep="last")

    from components.inference import _preprocess, _add_features, _to_ae_space, CONTINUOUS_COLS, CYCLIC_COLS

    processed, _raw = _preprocess(combined)
    engineered = _add_features(processed)

    rng = np.random.default_rng(42)
    n = len(engineered)
    perm = rng.permutation(n)
    train_n = int(n * 0.70)
    val_n   = int(n * 0.15)
    idx_train = perm[:train_n]
    idx_val   = perm[train_n:train_n + val_n]
    idx_test  = perm[train_n + val_n:]

    X_train = engineered.iloc[idx_train].reset_index(drop=True)
    X_val   = engineered.iloc[idx_val].reset_index(drop=True)
    X_test  = engineered.iloc[idx_test].reset_index(drop=True)

    #Train new IF
    iforest_new = IsolationForest(
        n_estimators=N_EST, contamination=CONTAMINATION,
        random_state=42, n_jobs=-1,).fit(X_train)

    # Training new AE on [0, 1]-scaled features 
    from sklearn.preprocessing import MinMaxScaler
    new_scaler = MinMaxScaler()
    X_train_scaled = X_train.copy()
    X_val_scaled   = X_val.copy()
    X_test_scaled  = X_test.copy()
    X_train_scaled[CONTINUOUS_COLS] = new_scaler.fit_transform(X_train[CONTINUOUS_COLS])
    X_val_scaled[CONTINUOUS_COLS]   = new_scaler.transform(X_val[CONTINUOUS_COLS])
    X_test_scaled[CONTINUOUS_COLS]  = new_scaler.transform(X_test[CONTINUOUS_COLS])
    for d in (X_train_scaled, X_val_scaled, X_test_scaled):
        d[CONTINUOUS_COLS] = d[CONTINUOUS_COLS].clip(0, 1)
        for col in CYCLIC_COLS:
            d[col] = (d[col] + 1) / 2

    feature_cols = X_train_scaled.columns.tolist()
    input_dim = len(feature_cols)

    torch.manual_seed(42)
    ae_new = Autoencoder(input_dim, HIDDEN, LATENT)
    optimizer = torch.optim.Adam(ae_new.parameters(), lr=LR)
    criterion = nn.MSELoss()

    Xt = torch.tensor(X_train_scaled[feature_cols].values.astype("float32"))
    Xv = torch.tensor(X_val_scaled[feature_cols].values.astype("float32"))

    best_val = float("inf")
    best_state = {k: v.clone() for k, v in ae_new.state_dict().items()}
    for epoch in range(EPOCHS):
        ae_new.train()
        perm = torch.randperm(len(Xt))
        for i in range(0, len(Xt), BATCH):
            batch = Xt[perm[i:i+BATCH]]
            optimizer.zero_grad()
            loss = criterion(ae_new(batch), batch)
            loss.backward()
            optimizer.step()
        ae_new.eval()
        with torch.no_grad():
            v = criterion(ae_new(Xv), Xv).item()
        if v < best_val - 1e-6:
            best_val = v
            best_state = {k: t.clone() for k, t in ae_new.state_dict().items()}
    ae_new.load_state_dict(best_state)

    with torch.no_grad():
        val_err = ((Xv - ae_new(Xv)) ** 2).mean(dim=1).numpy()
    ae_threshold = float(np.percentile(val_err, 95))

    test_metrics_new = _evaluate_on_synthetic(iforest_new, ae_new, ae_threshold,
                                              new_scaler, feature_cols, X_test, X_test_scaled)
    test_metrics_old = _evaluate_existing(X_test, X_test_scaled)

    #Save
    joblib.dump(iforest_new, STAGING / "isolation_forest.pkl")
    torch.save(ae_new.state_dict(), STAGING / "autoencoder.pth")
    joblib.dump({
        "input_dim": input_dim, "hidden_dim": HIDDEN, "latent_dim": LATENT,
        "threshold": ae_threshold, "feature_columns": feature_cols,
    }, STAGING / "autoencoder_metadata.pkl")
    joblib.dump(new_scaler, STAGING / "autoen_minmax_scaler.pkl")

    return {"training_samples": train_n,
            "val_samples":      val_n,
            "test_samples":     len(X_test),
            "old_if": test_metrics_old["if"],
            "new_if": test_metrics_new["if"],
            "old_ae": test_metrics_old["ae"],
            "new_ae": test_metrics_new["ae"], }
    
def _evaluate_existing(X_test: pd.DataFrame, X_test_scaled: pd.DataFrame) -> dict:
    """Score the test set with the *current production* models."""
    iforest_old = joblib.load(MODEL_DIR / "isolation_forest.pkl")
    meta = joblib.load(MODEL_DIR / "autoencoder_metadata.pkl")

    # Align cols for the old IF
    if_cols = list(iforest_old.feature_names_in_)
    aligned_if = X_test.copy()
    for c in if_cols:
        if c not in aligned_if.columns:
            aligned_if[c] = 0
    aligned_if = aligned_if[if_cols]
    if_score = -iforest_old.score_samples(aligned_if)

    ae_old = Autoencoder(meta["input_dim"], meta["hidden_dim"], meta["latent_dim"])
    ae_old.load_state_dict(torch.load(MODEL_DIR / "autoencoder.pth", map_location="cpu", weights_only=True))
    ae_old.eval()

    aligned_ae = X_test_scaled.copy()
    for c in meta["feature_columns"]:
        if c not in aligned_ae.columns:
            aligned_ae[c] = 0
    aligned_ae = aligned_ae[meta["feature_columns"]]
    x = torch.tensor(aligned_ae.values.astype("float32"))
    with torch.no_grad():
        ae_err = ((x - ae_old(x)) ** 2).mean(dim=1).numpy()

    return _scores_to_metrics(X_test, if_score, ae_err, meta["threshold"], iforest_old)

def _evaluate_on_synthetic(iforest_new, ae_new, ae_threshold, scaler, feature_cols, X_test, X_test_scaled) -> dict:
    if_score = -iforest_new.score_samples(X_test[iforest_new.feature_names_in_])
    x = torch.tensor(X_test_scaled[feature_cols].values.astype("float32"))
    with torch.no_grad():
        ae_err = ((x - ae_new(x)) ** 2).mean(dim=1).numpy()
    return _scores_to_metrics(X_test, if_score, ae_err, ae_threshold, iforest_new)

def _scores_to_metrics(X_test, if_score, ae_err, ae_threshold, iforest_model) -> dict:
    if_pred = (iforest_model.predict(X_test[iforest_model.feature_names_in_]) == -1).astype(int)
    ae_pred = (ae_err > ae_threshold).astype(int)
    return {
        "if": {
            "flagged_pct":  float(if_pred.mean() * 100),
            "score_mean":   float(if_score.mean()),
            "score_std":    float(if_score.std()),},
        "ae": {"flagged_pct":  float(ae_pred.mean() * 100),
               "error_mean":   float(ae_err.mean()),
               "threshold":    float(ae_threshold),},}
def commit() -> str:
    if not STAGING.exists():
        raise FileNotFoundError("Nothing in staging to commit.")

    BACKUPS.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_dir = BACKUPS / f"backup_{timestamp}"
    backup_dir.mkdir()

    for fname in ["isolation_forest.pkl", "autoencoder.pth", "autoencoder_metadata.pkl"]:
        src = MODEL_DIR / fname
        if src.exists():
            shutil.copy2(src, backup_dir / fname)
    src_scaler = SCALER_DIR / "autoen_minmax_scaler.pkl"
    if src_scaler.exists():
        shutil.copy2(src_scaler, backup_dir / "autoen_minmax_scaler.pkl")

    # Move staging
    for fname in ["isolation_forest.pkl", "autoencoder.pth", "autoencoder_metadata.pkl"]:
        shutil.move(str(STAGING / fname), str(MODEL_DIR / fname))
    shutil.move(str(STAGING / "autoen_minmax_scaler.pkl"), str(SCALER_DIR / "autoen_minmax_scaler.pkl"))

    STAGING.rmdir()
    return str(backup_dir)

def discard() -> None:
    if STAGING.exists():
        shutil.rmtree(STAGING)