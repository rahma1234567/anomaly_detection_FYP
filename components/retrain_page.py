import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.append(str(Path(__file__).resolve().parents[1]))
from logger import log_info, log_error  # noqa: E402

from components.inference import REQUIRED_COLUMNS  # noqa: E402
from retrain.pipeline import retrain, commit, discard  # noqa: E402


def render(username: str) -> None:
    st.header("Retrain models")
    st.write(
        "Upload new transaction data to retrain both models on the combined dataset. "
        "After retraining, compare the new and old models' behaviour and decide whether to deploy.")

    uploaded = st.file_uploader("New training CSV", type=["csv"], key="retrain_upload")
    if uploaded is None:
        with st.expander("Required columns"):
            st.write(REQUIRED_COLUMNS)
        return

    try:
        new_data = pd.read_csv(uploaded)
    except Exception as e:
        st.error(f"Couldn't read CSV: {e}")
        return

    missing = [c for c in REQUIRED_COLUMNS if c not in new_data.columns]
    if missing:
        st.error(f"Missing required columns: {missing}")
        return

    st.success(f"Loaded {uploaded.name}: {len(new_data):,} rows")

    if st.button("Retrain models", type="primary"):
        with st.spinner("Retraining... this can take 1-2 minutes."):
            try:
                report = retrain(new_data)
            except Exception as e:
                st.error(f"Retraining failed: {e}")
                log_error(f"[{username}] retraining failed: {e}")
                return
        st.session_state["retrain_report"] = report
        log_info(f"[{username}] retrained models on {len(new_data)} new rows")

    report = st.session_state.get("retrain_report")
    if report is None:
        return

    st.markdown("---")
    st.subheader("Comparison: production vs candidate")

    c1, c2, c3 = st.columns(3)
    c1.metric("Training samples", f"{report['training_samples']:,}")
    c2.metric("Validation samples", f"{report['val_samples']:,}")
    c3.metric("Test samples", f"{report['test_samples']:,}")

    st.markdown("### Isolation Forest")
    cols = st.columns(2)
    old_pct = report["old_if"]["flagged_pct"]
    new_pct = report["new_if"]["flagged_pct"]
    cols[0].metric("Production flag rate", f"{old_pct:.1f}%")
    cols[1].metric("Candidate flag rate", f"{new_pct:.1f}%", delta=f"{new_pct - old_pct:+.1f}%")

    st.markdown("### Autoencoder")
    cols = st.columns(3)
    cols[0].metric("Production flag rate", f"{report['old_ae']['flagged_pct']:.1f}%")
    cols[1].metric("Candidate flag rate", f"{report['new_ae']['flagged_pct']:.1f}%",
                   delta=f"{report['new_ae']['flagged_pct'] - report['old_ae']['flagged_pct']:+.1f}%")
    cols[2].metric("Candidate threshold (P95)", f"{report['new_ae']['threshold']:.4f}")

    st.caption(
        "Both models target ~5% flag rate by design (contamination=0.05 for IF, "
        "P95 of validation reconstruction error for AE). Large divergence between "
        "production and candidate suggests the new data has different patterns than "
        "the original training set - investigate before deploying."
    )

    st.markdown("---")
    st.subheader("Decision")
    a, b = st.columns(2)
    if a.button("Deploy candidate models", type="primary"):
        try:
            backup = commit()
        except Exception as e:
            st.error(f"Commit failed: {e}")
            return
        st.success(f"Deployed. Old models backed up to: `{backup}`")
        log_info(f"[{username}] committed retrained models. Backup: {backup}")
        del st.session_state["retrain_report"]
        st.rerun()

    if b.button("Discard candidates"):
        discard()
        st.warning("Candidate models discarded. Production models unchanged.")
        log_info(f"[{username}] discarded retrained models")
        del st.session_state["retrain_report"]
        st.rerun()