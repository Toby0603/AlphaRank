import os
from pathlib import Path
import pandas as pd
import streamlit as st

st.set_page_config(page_title="AlphaRank", page_icon="📈", layout="wide")

BASE_DIR = Path(__file__).resolve().parent
DATA_FILES = {
    "Core": BASE_DIR / "results_core.csv",
    "FTSE": BASE_DIR / "results_ftse.csv",
    "US": BASE_DIR / "results_us.csv",
    "Europe": BASE_DIR / "results_europe.csv",
}
PERF_FILE = BASE_DIR / "performance_history.csv"

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
    st.caption("Private access. For informational use only.")

    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if st.button("Login", type="primary"):
        app_username = os.environ.get("APP_USERNAME")
        app_password = os.environ.get("APP_PASSWORD")

        if app_username and app_password:
            if username == app_username and password == app_password:
                st.session_state.logged_in = True
                st.session_state.username = username
                st.rerun()
            else:
                st.error("Invalid username or password.")
        else:
            st.error("Login is not configured. Add APP_USERNAME and APP_PASSWORD as environment variables.")

    return False

def score_row(row: pd.Series):
    needed = ["Predicted Prob Up (%)", "Accuracy (%)", "Precision (%)", "F1 Score (%)", "RSI_14"]
    if any(pd.isna(row.get(col)) for col in needed):
        return None

    rsi = row["RSI_14"]
    bonus = 10 if rsi < 30 else 5 if rsi < 40 else -10 if rsi > 70 else -5 if rsi > 60 else 0

    return round(
        0.40 * row["Predicted Prob Up (%)"]
        + 0.20 * row["Accuracy (%)"]
        + 0.25 * row["Precision (%)"]
        + 0.15 * row["F1 Score (%)"]
        + bonus,
        1,
    )

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
def load_results(path_str: str):
    path = Path(path_str)
    if not path.exists():
        return pd.DataFrame()

    df = pd.read_csv(path)
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in {path.name}: {', '.join(missing)}")

    numeric_cols = [
        "Latest Price ($)",
        "Daily Return",
        "Volatility",
        "Predicted Prob Up (%)",
        "Accuracy (%)",
        "Precision (%)",
        "Recall (%)",
        "F1 Score (%)",
        "RSI_14",
    ]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    if "Score" not in df.columns:
        df["Score"] = df.apply(score_row, axis=1)
    if "Rating" not in df.columns:
        df["Rating"] = df["Score"].apply(rating_from_score)

    df = df.sort_values(["Score", "Predicted Prob Up (%)"], ascending=False, na_position="last")
    return df

@st.cache_data(ttl=300)
def load_performance(path_str: str):
    path = Path(path_str)
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    for col in ["forward_return_5d_pct", "hit", "pick_prob_up", "pick_score"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    if "pick_date" in df.columns:
        df["pick_date"] = pd.to_datetime(df["pick_date"], errors="coerce")
    return df

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

    st.subheader("Filters")
    min_prob = st.slider("Minimum probability up (%)", 0, 100, 55)
    min_score = st.slider("Minimum score", 0, 100, 55)
    max_vol = st.slider("Maximum volatility", 0.0, 0.20, 0.08, 0.005)
    min_accuracy = st.slider("Minimum accuracy (%)", 0, 100, 50)
    min_precision = st.slider("Minimum precision (%)", 0, 100, 50)
    rsi_low, rsi_high = st.slider("RSI range", 0, 100, (0, 70))
    ratings = st.multiselect(
        "Ratings",
        ["Strong Buy", "Watchlist", "Neutral", "Avoid"],
        default=["Strong Buy", "Watchlist", "Neutral"],
    )
    search = st.text_input("Ticker contains", "")

    if st.button("Logout"):
        st.session_state.logged_in = False
        st.session_state.username = ""
        st.rerun()

try:
    df = load_results(str(DATA_FILES[market]))
    perf = load_performance(str(PERF_FILE))
except Exception as e:
    st.error(f"Could not load data: {e}")
    st.stop()

st.title("📈 AlphaRank Screener")
st.caption("Multi-market stock ranking dashboard with performance tracking. Educational use only. Not financial advice.")

if df.empty:
    st.warning("No results found for this market yet.")
    st.stop()

filtered = df.copy()
filtered = filtered[filtered["Predicted Prob Up (%)"] >= min_prob]
filtered = filtered[filtered["Score"] >= min_score]
filtered = filtered[filtered["Volatility"] <= max_vol]
filtered = filtered[filtered["Accuracy (%)"] >= min_accuracy]
filtered = filtered[filtered["Precision (%)"] >= min_precision]
filtered = filtered[(filtered["RSI_14"] >= rsi_low) & (filtered["RSI_14"] <= rsi_high)]
filtered = filtered[filtered["Rating"].isin(ratings)]
if search:
    filtered = filtered[filtered["Ticker"].astype(str).str.contains(search.upper(), case=False, na=False)]

c1, c2, c3, c4 = st.columns(4)
c1.metric("Stocks shown", len(filtered))
c2.metric("Avg probability up", metric_format(filtered["Predicted Prob Up (%)"].mean(), "%"))
c3.metric("Avg score", metric_format(filtered["Score"].mean()))
c4.metric("Strong buys", int((filtered["Rating"] == "Strong Buy").sum()))

st.subheader(f"Top opportunities — {market}")
top_cols = ["Ticker", "Predicted Prob Up (%)", "Score", "Rating", "Accuracy (%)", "Precision (%)", "RSI_14"]
st.dataframe(filtered.head(10)[top_cols], use_container_width=True, hide_index=True)

left, right = st.columns([1.3, 1])

with left:
    st.subheader("Full ranked list")
    display_cols = [
        "Ticker",
        "Latest Price ($)",
        "Predicted Prob Up (%)",
        "Score",
        "Rating",
        "Accuracy (%)",
        "Precision (%)",
        "F1 Score (%)",
        "RSI_14",
        "Volatility",
        "Top Features",
    ]
    st.dataframe(filtered[display_cols], use_container_width=True, hide_index=True)

with right:
    st.subheader("Ticker detail")
    tickers = filtered["Ticker"].tolist() if not filtered.empty else df["Ticker"].tolist()
    selected_ticker = st.selectbox("Choose a ticker", tickers, index=0 if tickers else None)
    if selected_ticker:
        row = df[df["Ticker"] == selected_ticker].iloc[0]
        st.metric("Probability up", metric_format(row["Predicted Prob Up (%)"], "%"))
        st.metric("Score", metric_format(row["Score"]))
        st.metric("Rating", row["Rating"])
        st.metric("Accuracy", metric_format(row["Accuracy (%)"], "%"))
        st.metric("Precision", metric_format(row["Precision (%)"], "%"))
        st.metric("F1", metric_format(row["F1 Score (%)"], "%"))
        st.metric("RSI 14", metric_format(row["RSI_14"]))
        st.metric("Volatility", metric_format(row["Volatility"], "%"))
        st.markdown("**Top features**")
        st.write(row.get("Top Features", ""))

st.markdown("---")
st.subheader("Performance tracking")

if perf.empty:
    st.info("No performance history yet. Run the backend refresh workflow a few times to build history.")
else:
    perf_market = perf[perf["market"] == market.lower()].copy()

    p1, p2, p3, p4 = st.columns(4)
    p1.metric("Tracked picks", len(perf_market))
    p2.metric("Hit rate", metric_format(perf_market["hit"].mean() * 100, "%"))
    p3.metric("Avg 5d return", metric_format(perf_market["forward_return_5d_pct"].mean(), "%"))
    p4.metric("Avg pick score", metric_format(perf_market["pick_score"].mean()))

    st.markdown("**Performance by pick date**")
    by_date = perf_market.groupby("pick_date", dropna=True).agg(
        picks=("Ticker", "count"),
        hit_rate=("hit", "mean"),
        avg_return_5d=("forward_return_5d_pct", "mean"),
    ).reset_index().sort_values("pick_date", ascending=False)
    if not by_date.empty:
        by_date["hit_rate"] = by_date["hit_rate"] * 100
        st.dataframe(by_date, use_container_width=True, hide_index=True)

    st.markdown("**Recent tracked picks**")
    recent_cols = ["pick_date", "market", "Ticker", "pick_prob_up", "pick_score", "start_close", "end_close_5d", "forward_return_5d_pct", "hit"]
    available = [c for c in recent_cols if c in perf_market.columns]
    st.dataframe(perf_market.sort_values("pick_date", ascending=False)[available], use_container_width=True, hide_index=True)

st.markdown("---")
st.caption("This dashboard is for ranking and research. It does not provide investment advice.")
