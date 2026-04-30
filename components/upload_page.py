"""
Upload page - takes a CSV, runs inference, shows metrics + flagged table.
"""
import hashlib
import pandas as pd
import streamlit as st

from components.inference import run_pipeline, validate_columns, REQUIRED_COLUMNS
from components.shared import threshold_slider, apply_threshold, log_flagged_once, risk_badge

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))
from logger import log_info, log_error  # noqa: E402


def render(model_choice: str, username: str) -> None:
    st.header("Upload & Detect")
    st.write("Upload a CSV of transactions to run anomaly detection.")

    uploaded = st.file_uploader("CSV file", type=["csv"], key="upload_csv")

    if uploaded is None:
        with st.expander("Required columns"):
            st.write(REQUIRED_COLUMNS)
        st.info("Upload a file to begin.")
        return

    # Read once per upload; cache by content hash
    file_bytes = uploaded.getvalue()
    file_hash  = hashlib.md5(file_bytes).hexdigest()[:12]

    cache_key = f"_pipeline_{file_hash}_{model_choice}"
    if cache_key not in st.session_state:
        try:
            data = pd.read_csv(uploaded)
        except Exception as e:
            st.error(f"Couldn't read CSV: {e}")
            log_error(f"CSV read failed for {uploaded.name}: {e}")
            return

        missing = validate_columns(data)
        if missing:
            st.error(f"Missing required columns: {missing}")
            log_error(f"Upload {uploaded.name} missing columns: {missing}")
            return

        log_info(f"[{username}] uploaded {uploaded.name} ({len(data)} rows)")

        with st.spinner(f"Running {model_choice}..."):
            try:
                results, score_col, default_thr = run_pipeline(data, model_choice)
            except Exception as e:
                st.error(f"Pipeline failed: {e}")
                log_error(f"Pipeline error for {uploaded.name}: {e}")
                return

        st.session_state[cache_key] = (results, score_col, default_thr)
        log_info(f"[{username}] pipeline OK: {model_choice} on {uploaded.name}")

    results, score_col, default_thr = st.session_state[cache_key]

    # Threshold slider in sidebar
    score_range = (float(results[score_col].min()), float(results[score_col].max()))
    threshold = threshold_slider(score_col, default_thr, score_range)

    # Apply threshold
    flagged_df = apply_threshold(results, score_col, threshold)
    st.session_state["current_results"] = flagged_df
    st.session_state["current_model"]   = model_choice
    st.session_state["current_score_col"] = score_col

    # Log anomalies (deduped by file+model+threshold)
    log_flagged_once(flagged_df, model_choice, username, threshold, file_hash)

    # ---------- Metrics ----------
    n_total    = len(flagged_df)
    n_anom     = int(flagged_df["is_anomaly"].sum())
    n_high     = int((flagged_df["risk_level"] == "High").sum())
    pct_anom   = n_anom / n_total * 100 if n_total else 0.0

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total transactions", n_total)
    c2.metric("Flagged anomalies", n_anom, delta=f"{pct_anom:.1f}%")
    c3.metric("High-risk", n_high)
    c4.metric("Active threshold", f"{threshold:.4f}")

    st.markdown("---")

    # ---------- Flagged transactions table ----------
    st.subheader(f"Flagged transactions ({n_anom})")
    if n_anom == 0:
        st.info("No anomalies detected at this threshold. Try lowering the slider.")
        return

    show = flagged_df[flagged_df["is_anomaly"] == 1].sort_values("risk_pct", ascending=False)
    show = show.assign(Risk=show["risk_level"].apply(risk_badge))

    display_cols = ["TransactionID", "AccountID", "TransactionDate",
                    "TransactionAmount", "Location", "AccountBalance",
                    "Risk", "risk_pct", score_col]
    st.dataframe(show[display_cols], use_container_width=True, height=420)

    # ---------- Download button ----------
    csv = show.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download flagged transactions (CSV)", csv,
        f"flagged_{model_choice.replace(' ', '_').lower()}_{file_hash}.csv",
        "text/csv",
    )