import os
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st
import yfinance as yf

st.set_page_config(page_title="AlphaRank", page_icon="📈", layout="wide")

BASE_DIR = Path(__file__).resolve().parent

DATA_FILES = {
    "Core": BASE_DIR / "results_core.csv",
    "FTSE": BASE_DIR / "results_ftse.csv",
    "US": BASE_DIR / "results_us.csv",
    "Europe": BASE_DIR / "results_europe.csv",
}

PERF_FILE = BASE_DIR / "performance_history.csv"
WEEKLY_FILE = BASE_DIR / "weekly_summary.csv"
BENCH_FILE = BASE_DIR / "benchmark_summary.csv"

REQUIRED_COLUMNS = [
    "Ticker",
    "Latest Price ($)",
    "Daily Return",
    "Volatility",
    "Predicted Prob Up (%)",
    "Accuracy (%)",
    "Precision (%)",
    "Recall (%)",
    "F1 Score (%)",
    "RSI_14",
    "Top Features",
]

def init_session():
    if "logged_in" not in st.session_state:
        st.session_state.logged_in = False
    if "username" not in st.session_state:
        st.session_state.username = ""

def check_login():
    init_session()
    if st.session_state.logged_in:
        return True

    st.title("AlphaRank Login")
    st.caption("Private access to a quantitative research dashboard. Not financial advice.")

    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if st.button("Login"):
        if username == os.environ.get("APP_USERNAME") and password == os.environ.get("APP_PASSWORD"):
            st.session_state.logged_in = True
            st.session_state.username = username
            st.rerun()
        else:
            st.error("Invalid login")

    return False

def score_row(row):
    if any(pd.isna(row.get(c)) for c in ["Predicted Prob Up (%)","Accuracy (%)","Precision (%)","F1 Score (%)","RSI_14"]):
        return None
    rsi = row["RSI_14"]
    bonus = 10 if rsi < 30 else 5 if rsi < 40 else -10 if rsi > 70 else -5 if rsi > 60 else 0
    return round(
        0.40*row["Predicted Prob Up (%)"]
        + 0.20*row["Accuracy (%)"]
        + 0.25*row["Precision (%)"]
        + 0.15*row["F1 Score (%)"]
        + bonus,
        1
    )

def rating_from_score(score):
    if pd.isna(score):
        return ""
    if score >= 70:
        return "Top Ranked"
    if score >= 60:
        return "Above Average"
    if score >= 50:
        return "Neutral"
    return "Below Average"

@st.cache_data(ttl=300)
def load_results(path):
    if not Path(path).exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    for col in REQUIRED_COLUMNS:
        if col not in df.columns:
            raise ValueError(f"Missing column {col}")
    for col in REQUIRED_COLUMNS[1:]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["Score"] = df.apply(score_row, axis=1)
    df["Rating"] = df["Score"].apply(rating_from_score)
    return df.sort_values("Score", ascending=False)

def generate_report(df):
    return df.sort_values("Score", ascending=False)

if not check_login():
    st.stop()

with st.sidebar:
    st.title("AlphaRank")
    st.write(f"Logged in as {st.session_state.username}")
    market = st.selectbox("Market", list(DATA_FILES.keys()))

st.title("📈 AlphaRank Screener")
st.caption("Quantitative stock ranking tool. Not financial advice.")

df = load_results(DATA_FILES[market])

if df.empty:
    st.warning("No data available")
    st.stop()

st.subheader("📊 Weekly Top Ranked Signals")
top = df.head(10)
st.dataframe(top, use_container_width=True)

report_df = generate_report(df)
csv = report_df.to_csv(index=False).encode("utf-8")

st.download_button(
    label="📄 Download CSV",
    data=csv,
    file_name="alpharank_report.csv",
    mime="text/csv"
)

st.markdown("---")
st.markdown("**Disclaimer**: This tool provides quantitative research only and is not financial advice.")
