import os
from pathlib import Path
import pandas as pd
import streamlit as st
import yfinance as yf
import altair as alt

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
    "Ticker","Latest Price ($)","Daily Return","Volatility","Predicted Prob Up (%)",
    "Accuracy (%)","Precision (%)","Recall (%)","F1 Score (%)","RSI_14","Top Features",
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
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    if st.button("Login", type="primary"):
        u = os.environ.get("APP_USERNAME")
        p = os.environ.get("APP_PASSWORD")
        if u and p and username == u and password == p:
            st.session_state.logged_in = True
            st.session_state.username = username
            st.rerun()
        else:
            st.error("Invalid username or password.")
    return False

def score_row(row):
    needed = ["Predicted Prob Up (%)","Accuracy (%)","Precision (%)","F1 Score (%)","RSI_14"]
    if any(pd.isna(row.get(col)) for col in needed):
        return None
    rsi = row["RSI_14"]
    bonus = 10 if rsi < 30 else 5 if rsi < 40 else -10 if rsi > 70 else -5 if rsi > 60 else 0
    return round(0.40*row["Predicted Prob Up (%)"] + 0.20*row["Accuracy (%)"] + 0.25*row["Precision (%)"] + 0.15*row["F1 Score (%)"] + bonus, 1)

def rating_from_score(score):
    if pd.isna(score):
        return ""
    if score >= 70:
        return "Strong Buy"
    if score >= 60:
        return "Watchlist"
    if score >= 50:
        return "Neutral"
    return "Avoid"

@st.cache_data(ttl=300)
def load_results(path_str):
    path = Path(path_str)
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    for col in ["Latest Price ($)","Daily Return","Volatility","Predicted Prob Up (%)","Accuracy (%)","Precision (%)","Recall (%)","F1 Score (%)","RSI_14"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    if "Score" not in df.columns:
        df["Score"] = df.apply(score_row, axis=1)
    if "Rating" not in df.columns:
        df["Rating"] = df["Score"].apply(rating_from_score)
    return df.sort_values(["Score","Predicted Prob Up (%)"], ascending=False)

@st.cache_data(ttl=300)
def load_csv(path_str, date_cols=None):
    path = Path(path_str)
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    for col in date_cols or []:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")
    return df

@st.cache_data(ttl=900)
def load_ticker_chart(ticker):
    data = yf.download(ticker, period="6mo", auto_adjust=False, progress=False, threads=False)
    if data.empty:
        return pd.DataFrame()
    data = data.reset_index()
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = [c[0] if c[0] else c[1] for c in data.columns]
    data = data.loc[:, ~pd.Index(data.columns).duplicated()]
    if "Date" in data.columns and "Close" in data.columns:
        data["Date"] = pd.to_datetime(data["Date"], errors="coerce")
        data["Close"] = pd.to_numeric(data["Close"], errors="coerce")
        data["MA50"] = data["Close"].rolling(50).mean()
        return data[["Date","Close","MA50"]].dropna(subset=["Date","Close"])
    return pd.DataFrame()

def metric_format(value, suffix=""):
    if pd.isna(value):
        return "-"
    return f"{value:.1f}{suffix}"

if not check_login():
    st.stop()

with st.sidebar:
    st.title("AlphaRank")
    st.write(f"Logged in as: {st.session_state.username}")
    market = st.selectbox("Market", list(DATA_FILES.keys()), index=0)
    min_prob = st.slider("Minimum probability up (%)", 0, 100, 55)
    min_score = st.slider("Minimum score", 0, 100, 55)
    max_vol = st.slider("Maximum volatility", 0.0, 0.20, 0.08, 0.005)
    strong_only = st.toggle("Strong Buy only", value=False)
    if st.button("Logout"):
        st.session_state.logged_in = False
        st.session_state.username = ""
        st.rerun()

df = load_results(str(DATA_FILES[market]))
perf = load_csv(str(PERF_FILE), ["pick_date"])
weekly = load_csv(str(WEEKLY_FILE), ["week_start"])
bench = load_csv(str(BENCH_FILE), ["week_start"])

st.title("📈 AlphaRank Screener")
if df.empty:
    st.warning("No results found for this market yet.")
    st.stop()

filtered = df.copy()
filtered = filtered[filtered["Predicted Prob Up (%)"] >= min_prob]
filtered = filtered[filtered["Score"] >= min_score]
filtered = filtered[filtered["Volatility"] <= max_vol]
if strong_only:
    filtered = filtered[filtered["Rating"] == "Strong Buy"]

c1, c2, c3, c4 = st.columns(4)
c1.metric("Stocks shown", len(filtered))
c2.metric("Avg probability up", metric_format(filtered["Predicted Prob Up (%)"].mean(), "%"))
c3.metric("Avg score", metric_format(filtered["Score"].mean()))
c4.metric("Strong buys", int((filtered["Rating"] == "Strong Buy").sum()))

tabs = st.tabs(["Screener", "Charts", "Weekly Summary", "Benchmarking", "Weekly Output"])

with tabs[0]:
    st.dataframe(filtered.head(10)[["Ticker","Predicted Prob Up (%)","Score","Rating","Accuracy (%)","Precision (%)","RSI_14"]], use_container_width=True, hide_index=True)

with tabs[1]:
    tickers = filtered["Ticker"].tolist() if not filtered.empty else df["Ticker"].tolist()
    chart_ticker = st.selectbox("Ticker for chart", tickers, index=0 if tickers else None)
    if chart_ticker:
        chart_df = load_ticker_chart(chart_ticker)
        if chart_df.empty:
            st.info("No chart data available.")
        else:
            chart_long = chart_df.melt("Date", value_vars=["Close","MA50"], var_name="Series", value_name="Value")
            st.altair_chart(
                alt.Chart(chart_long).mark_line().encode(x="Date:T", y="Value:Q", color="Series:N"),
                use_container_width=True
            )

with tabs[2]:
    wk = weekly[weekly["market"] == market.lower()].copy() if not weekly.empty else pd.DataFrame()
    if wk.empty:
        st.info("No weekly summary yet.")
    else:
        st.dataframe(wk.sort_values("week_start", ascending=False), use_container_width=True, hide_index=True)

with tabs[3]:
    b = bench[bench["market"] == market.lower()].copy() if not bench.empty else pd.DataFrame()
    if b.empty:
        st.info("No benchmark summary yet.")
    else:
        st.dataframe(b.sort_values("week_start", ascending=False), use_container_width=True, hide_index=True)
        st.altair_chart(
            alt.Chart(b).mark_line(point=True).encode(x="week_start:T", y="alpha_vs_benchmark_pct:Q"),
            use_container_width=True
        )

with tabs[4]:
    wk = weekly[weekly["market"] == market.lower()].copy() if not weekly.empty else pd.DataFrame()
    if wk.empty:
        st.info("No weekly output yet.")
    else:
        latest_week = wk["week_start"].max()
        latest = wk[wk["week_start"] == latest_week]
        st.dataframe(latest, use_container_width=True, hide_index=True)
        st.download_button("Download latest weekly output", latest.to_csv(index=False).encode("utf-8"), file_name=f"alpharank_{market.lower()}_weekly_output.csv", mime="text/csv")

st.markdown("---")
st.caption("This dashboard is for ranking and research. It does not provide investment advice.")
