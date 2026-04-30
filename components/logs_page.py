import os
from pathlib import Path
import pandas as pd
import streamlit as st

LOG_DIR = Path("logs")

def render() -> None:
    st.header("Audit logs")

    if not LOG_DIR.exists():
        st.info("No logs")
        return

    log_type = st.radio("Log type", ["System", "Anomalies"], horizontal=True)
    prefix = "system" if log_type == "System" else "anomalies"

    # Find available dates from filenames
    available = sorted({
        f.stem.split("_", 1)[1]
        for f in LOG_DIR.glob(f"{prefix}_*.log")
    }, reverse=True)

    if not available:
        st.info(f"No {log_type.lower()} logs yet.")
        return

    selected = st.selectbox("Date", available)
    log_file = LOG_DIR / f"{prefix}_{selected}.log"

    if not log_file.exists():
        st.info(f"No {log_type.lower()} logs for {selected}.")
        return

    rows = []
    with open(log_file, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split(" | ", 2)
            if len(parts) == 3:
                rows.append({"Timestamp": parts[0], "Level": parts[1], "Message": parts[2]})

    if not rows:
        st.info("Log file is empty.")
        return

    df = pd.DataFrame(rows)

    # Filters
    fc1, fc2 = st.columns([1, 2])
    levels = df["Level"].unique().tolist()
    level_filter = fc1.multiselect("Level", levels, default=levels)
    search = fc2.text_input("Search", "")

    filt = df[df["Level"].isin(level_filter)]
    if search:
        filt = filt[filt["Message"].str.contains(search, case=False, na=False)]

    st.write(f"Showing {len(filt)} of {len(df)} entries")
    st.dataframe(filt, use_container_width=True, height=420)

    csv = filt.to_csv(index=False).encode("utf-8")
    st.download_button("Download as CSV", csv, f"{prefix}_{selected}.csv", "text/csv")