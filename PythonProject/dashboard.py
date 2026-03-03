import streamlit as st
import pandas as pd
import plotly.express as px
import json
import os
import time

# ── PAGE CONFIGURATION ──
st.set_page_config(page_title="Planet Factory: Hive Command", layout="wide", page_icon="🌌")
st.title("🌌 Planet Factory: Hive Command Dashboard")

LOG_FILE = "hive_log.json"


def load_data():
    if not os.path.exists(LOG_FILE):
        return pd.DataFrame()
    with open(LOG_FILE, "r") as f:
        try:
            data = json.load(f)
            return pd.DataFrame(data)
        except json.JSONDecodeError:
            return pd.DataFrame()


# ── AUTO-REFRESH TOGGLE ──
col1, col2 = st.columns([8, 2])
with col2:
    live_update = st.toggle("🔴 Live Update (Auto-refresh)")

df = load_data()

if df.empty:
    st.warning(f"Waiting for telemetry... Run your `ucf.py` script to generate {LOG_FILE}.")
else:
    # ── METRICS OVERVIEW ──
    st.markdown("### 🏭 Global Hive Metrics")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total Cycles Executed", df["cycle"].max())
    m2.metric("Active Machines", df["machine"].nunique())

    safe_pct = (df["safe"].sum() / len(df)) * 100
    m3.metric("Safety Compliance", f"{safe_pct:.1f}%")

    drifting = len(df[df["mood"] == "drifting"])
    m4.metric("Quantum Drift Events", drifting)

    st.markdown("---")

    # ── VISUALIZATIONS ──
    row1_col1, row1_col2 = st.columns(2)

    with row1_col1:
        st.markdown("#### 🛑 Safety Governor Interventions")
        # Filter for blocked actions
        blocks = df[df["safe"] == False]
        if not blocks.empty:
            fig_safety = px.histogram(
                blocks, x="machine", color="reason",
                title="Blocks per Machine by Reason",
                color_discrete_sequence=px.colors.qualitative.Pastel
            )
            st.plotly_chart(fig_safety, use_container_width=True)
        else:
            st.success("No safety violations detected! Governor is clear.")

    with row1_col2:
        st.markdown("#### ⚙️ Action Distribution")
        fig_actions = px.histogram(
            df, x="action", color="machine",
            title="Frequency of Executed Actions",
            barmode="group"
        )
        st.plotly_chart(fig_actions, use_container_width=True)

    row2_col1, row2_col2 = st.columns(2)

    with row2_col1:
        st.markdown("#### 🧠 Quantum Sentiment Timeline")
        fig_mood = px.scatter(
            df, x="cycle", y="machine", color="mood", symbol="mood",
            title="Machine Mood Drift over Cycles",
            color_discrete_map={"stable": "green", "drifting": "red"}
        )
        fig_mood.update_traces(marker=dict(size=12))
        st.plotly_chart(fig_mood, use_container_width=True)

    with row2_col2:
        st.markdown("#### 📜 Raw Telemetry Log (Latest)")
        st.dataframe(df.sort_values(by="cycle", ascending=False).head(15), use_container_width=True)

# Handle live updating
if live_update:
    time.sleep(2)
    st.rerun()