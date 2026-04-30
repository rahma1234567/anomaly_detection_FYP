"""
Shared UI helpers used across multiple pages.
"""
import hashlib
from typing import Optional

import pandas as pd
import streamlit as st

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))
from logger import log_anomaly  # noqa: E402


# ---------------------------------------------------------------
# Threshold slider - lets the user override the model's default
# ---------------------------------------------------------------
def threshold_slider(score_col: str, default: float, score_range: tuple[float, float]) -> float:
    """Display a sidebar slider for the score threshold, return the chosen value."""
    lo, hi = score_range
    if hi <= lo:
        return default

    label = "Anomaly threshold" if score_col == "anomaly_score" else "Reconstruction-error threshold"
    return st.sidebar.slider(
        label, min_value=float(lo), max_value=float(hi),
        value=float(min(max(default, lo), hi)),
        step=(hi - lo) / 200,
        help="Higher = stricter (fewer alerts, higher precision). "
             "Lower = looser (more alerts, higher recall).",
    )


# ---------------------------------------------------------------
# Risk percentage - re-flag using the chosen threshold
# ---------------------------------------------------------------
def apply_threshold(df: pd.DataFrame, score_col: str, threshold: float) -> pd.DataFrame:
    """Re-flag rows using the threshold and compute a relative risk percentage."""
    df = df.copy()
    df["is_anomaly"] = (df[score_col] > threshold).astype(int)

    # Risk percentage: how far above threshold the score is, capped at the
    # observed max. Below threshold = 0%. Anchored to threshold so the slider
    # has a visible effect on the percentage.
    score = df[score_col].to_numpy()
    above = score - threshold
    max_above = above.max() if (above > 0).any() else 1.0
    df["risk_pct"] = ((above / max_above).clip(0, 1) * 100).round(1)
    df["risk_level"] = df["risk_pct"].apply(_risk_level)
    return df


def _risk_level(pct: float) -> str:
    if pct >= 75: return "High"
    if pct >= 40: return "Medium"
    if pct >  0:  return "Low"
    return "None"


# ---------------------------------------------------------------
# Anomaly logging with dedupe per (file, model, threshold)
# ---------------------------------------------------------------
def log_flagged_once(df: pd.DataFrame, model: str, username: str,
                     threshold: float, file_signature: Optional[str]) -> None:
    """
    Log each flagged transaction exactly once per unique
    (file, model, threshold) combination. Re-uploading the same file at the
    same threshold won't duplicate-log; changing threshold will re-log.
    """
    if file_signature is None:
        # Generate a stable signature from the data itself
        file_signature = hashlib.md5(
            pd.util.hash_pandas_object(df["TransactionID"], index=False).values.tobytes()
        ).hexdigest()[:8]

    log_key = f"_logged_{file_signature}_{model}_{threshold:.6f}"
    if log_key in st.session_state:
        return

    flagged = df[df["is_anomaly"] == 1]
    for _, row in flagged.iterrows():
        log_anomaly(
            transaction_id=str(row["TransactionID"]),
            account_id=str(row["AccountID"]),
            amount=float(row["TransactionAmount"]),
            location=str(row["Location"]),
            risk_pct=float(row["risk_pct"]),
            model=model,
            triggered_by=username,
        )
    st.session_state[log_key] = True


# ---------------------------------------------------------------
# Risk badge styling for tables
# ---------------------------------------------------------------
RISK_EMOJI = {"High": "🟥", "Medium": "🟨", "Low": "🟩", "None": "⬜"}

def risk_badge(level: str) -> str:
    return f"{RISK_EMOJI.get(level, '⬜')} {level}"