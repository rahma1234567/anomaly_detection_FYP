from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st
import torch

from components.inference import (
    Autoencoder, AE_MODEL_PATH, AE_META_PATH,
    _preprocess, _add_features, _to_ae_space,
)
from components.shared import risk_badge


def render() -> None:
    st.header("Investigate a flagged transaction")

    if "current_results" not in st.session_state:
        st.info("Upload a CSV on the Upload page first, then come back here.")
        return

    df         = st.session_state["current_results"]
    score_col  = st.session_state["current_score_col"]
    model_name = st.session_state["current_model"]

    flagged = df[df["is_anomaly"] == 1].sort_values("risk_pct", ascending=False)
    if len(flagged) == 0:
        st.info("No anomalies flagged at the current threshold.")
        return

    # Picker
    options = [
        f"{row.TransactionID}  -  £{row.TransactionAmount:.2f}  -  {row.risk_level} risk ({row.risk_pct:.1f}%)"
        for row in flagged.itertuples()
    ]
    choice = st.selectbox("Select a transaction", options, index=0)
    idx = flagged.index[options.index(choice)]
    tx = df.loc[idx]

    #Header card
    h1, h2, h3 = st.columns([2, 2, 1])
    h1.markdown(f"### Transaction `{tx['TransactionID']}`")
    h1.write(f"**Account:** {tx['AccountID']}  \n"
             f"**Date:** {tx['TransactionDate']}  \n"
             f"**Location:** {tx['Location']}  \n"
             f"**Channel:** {tx['Channel']}")
    h2.write(f"**Amount:** £{tx['TransactionAmount']:.2f}  \n"
             f"**Balance:** £{tx['AccountBalance']:.2f}  \n"
             f"**Login attempts:** {int(tx['LoginAttempts'])}  \n"
             f"**Duration:** {int(tx['TransactionDuration'])}s")
    h3.metric("Risk", risk_badge(tx["risk_level"]))
    h3.metric("Score", f"{tx[score_col]:.4f}")

    st.markdown("---")

    #Heuristic reasons
    st.subheader("Why this was flagged")
    reasons = _heuristic_reasons(tx, df)
    if reasons:
        for r in reasons:
            st.write(f"- {r}")
    else:
        st.write("- No single rule explains this — flagged on combined signal.")

    #Model-specific explanation
    st.markdown("---")
    if model_name == "Autoencoder":
        st.subheader("Per-feature reconstruction error")
        _show_ae_per_feature_error(tx, df)
    else:
        st.subheader("Feature values vs population")
        _show_if_feature_distribution(tx, df)


def _heuristic_reasons(tx: pd.Series, df: pd.DataFrame) -> list[str]:
    reasons = []

    p95_amount = df["TransactionAmount"].quantile(0.95)
    if tx["TransactionAmount"] > p95_amount:
        reasons.append(f"Amount £{tx['TransactionAmount']:.2f} is in the top 5% of all transactions (>£{p95_amount:.2f})")

    if tx["AccountBalance"] > 0 and tx["TransactionAmount"] / tx["AccountBalance"] > 0.5:
        ratio = tx["TransactionAmount"] / tx["AccountBalance"] * 100
        reasons.append(f"Transaction uses {ratio:.0f}% of account balance")

    if tx["LoginAttempts"] >= 3:
        reasons.append(f"{int(tx['LoginAttempts'])} login attempts (typical = 1)")

    if tx["TransactionDuration"] < 10:
        reasons.append(f"Very fast transaction ({int(tx['TransactionDuration'])} seconds)")
    elif tx["TransactionDuration"] > df["TransactionDuration"].quantile(0.95):
        reasons.append(f"Unusually long duration ({int(tx['TransactionDuration'])} seconds)")

    return reasons


def _show_ae_per_feature_error(tx: pd.Series, df: pd.DataFrame) -> None:
    """
    Re-run autoencoder on this transaction (and its peers) and show which
    features had the largest reconstruction error.
    """
    # Reuse the same preprocessing for the *entire* batch, then pick out the row.
    # This guarantees the AE input matches training-time scaling.
    raw_for_pipeline = df[
        ["TransactionID", "AccountID", "TransactionAmount", "TransactionDate",
         "TransactionType", "Location", "DeviceID", "IP Address", "MerchantID",
         "Channel", "CustomerAge", "CustomerOccupation", "TransactionDuration",
         "LoginAttempts", "AccountBalance", "PreviousTransactionDate"]
    ].copy() if "DeviceID" in df.columns else None

    if raw_for_pipeline is None:
        st.write("Cannot compute per-feature errors (raw data missing).")
        return

    processed, raw = _preprocess(raw_for_pipeline)
    engineered = _add_features(processed)
    scaled = _to_ae_space(engineered)

    meta = joblib.load(AE_META_PATH)
    aligned = scaled.copy()
    for col in meta["feature_columns"]:
        if col not in aligned.columns:
            aligned[col] = 0
    aligned = aligned[meta["feature_columns"]]

    model = Autoencoder(meta["input_dim"], meta["hidden_dim"], meta["latent_dim"])
    model.load_state_dict(torch.load(AE_MODEL_PATH, map_location="cpu", weights_only=True))
    model.eval()

    x = torch.tensor(aligned.values.astype("float32"), dtype=torch.float32)
    with torch.no_grad():
        recon = model(x).numpy()
    per_feat_err = (aligned.values - recon) ** 2

    # Find which row in `aligned` corresponds to the chosen transaction
    target_id = tx["TransactionID"]
    raw_idx = raw.index[raw["TransactionID"] == target_id]
    if len(raw_idx) == 0:
        st.write("Could not locate this transaction in the model input.")
        return
    row_pos = raw_idx[0]

    errors = pd.Series(per_feat_err[row_pos], index=meta["feature_columns"])
    top = errors.sort_values(ascending=False).head(8)

    chart_df = pd.DataFrame({"feature": top.index, "reconstruction_error": top.values})
    st.bar_chart(chart_df, x="feature", y="reconstruction_error", height=260)
    st.caption("Higher bars = features the model couldn't reconstruct well = strongest anomaly signals.")


def _show_if_feature_distribution(tx: pd.Series, df: pd.DataFrame) -> None:
    """
    For each numeric feature, show how this transaction's value compares to
    the population (z-score). Highlight features where |z| > 2.
    """
    numeric_cols = ["TransactionAmount", "AccountBalance", "LoginAttempts",
                    "TransactionDuration", "CustomerAge"]
    rows = []
    for col in numeric_cols:
        if col not in df.columns:
            continue
        mean = df[col].mean()
        std  = df[col].std()
        if std == 0 or pd.isna(std):
            z = 0.0
        else:
            z = (tx[col] - mean) / std
        rows.append({
            "feature": col,
            "value": tx[col],
            "population_mean": round(mean, 2),
            "z_score": round(z, 2),
            "unusual": "⚠️" if abs(z) > 2 else "",
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    st.caption("z_score above 2 (or below -2) means the value is unusual versus other transactions in this batch.")