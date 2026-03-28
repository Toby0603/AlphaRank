import json
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


def load_users():
    raw = os.environ.get("APP_USERS_JSON", "").strip()

    if raw:
        try:
            users = json.loads(raw)
            if isinstance(users, dict) and users:
                cleaned = {}
                for username, password in users.items():
                    if username is None or password is None:
                        continue
                    cleaned[str(username).strip()] = str(password)
                if cleaned:
                    return cleaned, None
            return None, "APP_USERS_JSON must be a non-empty JSON object."
        except json.JSONDecodeError:
            return None, "APP_USERS_JSON is not valid JSON."

    username = os.environ.get("APP_USERNAME", "").strip()
    password = os.environ.get("APP_PASSWORD", "")
    if username and password:
        return {username: password}, None

    return None, "No login credentials are configured."


def check_login():
    init_session()

    if st.session_state.logged_in:
        return True

    users, error = load_users()

    st.title("AlphaRank Login")
    st.caption(
        "Private access to a quantitative research dashboard. "
        "This application is for informational use only."
    )

    if error:
        st.error(error)
        return False

    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if st.button("Login", type="primary"):
        entered_user = username.strip()
        if entered_user in users and password == users[entered_user]:
            st.session_state.logged_in = True
            st.session_state.username = entered_user
            st.rerun()
        else:
            st.error("Invalid username or password.")

    return False


def score_row(row: pd.Series):
    needed = [
        "Predicted Prob Up (%)",
        "Accuracy (%)",
        "Precision (%)",
        "F1 Score (%)",
        "RSI_14",
    ]
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
        return "Top Ranked"
    if score >= 60:
        return "Above Average"
    if score >= 50:
        return "Neutral"
    return "Below Average"


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

    return df.sort_values(["Score", "Predicted Prob Up (%)"], ascending=False, na_position="last")


@st.cache_data(ttl=300)
def load_csv(path_str: str, date_cols=None):
    path = Path(path_str)
    if not path.exists():
        return pd.DataFrame()

    df = pd.read_csv(path)
    for col in date_cols or []:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")
    return df


@st.cache_data(ttl=900)
def load_ticker_chart(ticker: str):
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
        return data[["Date", "Close", "MA50"]].dropna(subset=["Date", "Close"])

    return pd.DataFrame()


def metric_format(value, suffix=""):
    if pd.isna(value):
        return "-"
    return f"{value:.1f}{suffix}"


def generate_report(df: pd.DataFrame):
    cols = [
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
    available = [c for c in cols if c in df.columns]
    return df[available].sort_values("Score", ascending=False)


if not check_login():
    st.stop()

with st.sidebar:
    st.title("AlphaRank")
    st.write(f"Logged in as: {st.session_state.username}")

    market = st.selectbox("Market", list(DATA_FILES.keys()), index=0)

    st.subheader("Filters")
    min_prob = st.slider("Minimum model probability (%)", 0, 100, 55)
    min_score = st.slider("Minimum score", 0, 100, 55)
    max_vol = st.slider("Maximum volatility", 0.0, 0.20, 0.08, 0.005)
    min_accuracy = st.slider("Minimum accuracy (%)", 0, 100, 50)
    min_precision = st.slider("Minimum precision (%)", 0, 100, 50)
    rsi_low, rsi_high = st.slider("RSI range", 0, 100, (0, 70))
    ratings = st.multiselect(
        "Rating bands",
        ["Top Ranked", "Above Average", "Neutral", "Below Average"],
        default=["Top Ranked", "Above Average", "Neutral"],
    )
    top_only = st.toggle("Top Ranked only", value=False)
    search = st.text_input("Ticker contains", "")

    if st.button("Logout"):
        st.session_state.logged_in = False
        st.session_state.username = ""
        st.rerun()

try:
    df = load_results(str(DATA_FILES[market]))
    perf = load_csv(str(PERF_FILE), ["pick_date"])
    weekly = load_csv(str(WEEKLY_FILE), ["week_start"])
    bench = load_csv(str(BENCH_FILE), ["week_start"])
except Exception as e:
    st.error(f"Could not load data: {e}")
    st.stop()

st.title("📈 AlphaRank Screener")
st.caption(
    "Quantitative stock research and ranking tool. "
    "This application provides data-driven insights for informational purposes only "
    "and does not constitute financial advice or investment recommendations."
)
st.caption(
    "Model outputs are based on historical data patterns and do not account for "
    "individual circumstances, market conditions, or future events."
)

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

if top_only:
    filtered = filtered[filtered["Rating"] == "Top Ranked"]

if search:
    filtered = filtered[
        filtered["Ticker"].astype(str).str.contains(search.upper(), case=False, na=False)
    ]

c1, c2, c3, c4 = st.columns(4)
c1.metric("Stocks shown", len(filtered))
c2.metric("Avg model probability", metric_format(filtered["Predicted Prob Up (%)"].mean(), "%"))
c3.metric("Avg score", metric_format(filtered["Score"].mean()))
c4.metric("Top ranked", int((filtered["Rating"] == "Top Ranked").sum()))

tabs = st.tabs(
    [
        "Screener",
        "Charts",
        "Historical Signal Performance",
        "Weekly Summary",
        "Benchmarking",
        "Weekly Output",
    ]
)

with tabs[0]:
    st.subheader(f"📊 Weekly Top Ranked Signals — {market}")
    st.caption("Top ranked signals based on current model scoring and active filters.")

    top_weekly = filtered.sort_values("Score", ascending=False).head(10)
    weekly_cols = [
        "Ticker",
        "Predicted Prob Up (%)",
        "Score",
        "Rating",
        "Accuracy (%)",
        "Precision (%)",
        "RSI_14",
    ]
    available_weekly_cols = [c for c in weekly_cols if c in top_weekly.columns]
    if not top_weekly.empty:
        st.dataframe(top_weekly[available_weekly_cols], use_container_width=True, hide_index=True)
    else:
        st.info("No qualifying signals this week based on current filters.")

    report_df = generate_report(filtered)
    csv = report_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="📄 Download ranked report (CSV)",
        data=csv,
        file_name=f"alpharank_{market.lower()}_report.csv",
        mime="text/csv",
    )

    st.markdown("---")
    left, right = st.columns([1.4, 1])

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
        available_display_cols = [c for c in display_cols if c in filtered.columns]
        st.dataframe(filtered[available_display_cols], use_container_width=True, hide_index=True)

    with right:
        st.subheader("Ticker detail")
        tickers = filtered["Ticker"].tolist() if not filtered.empty else df["Ticker"].tolist()
        selected_ticker = st.selectbox(
            "Choose a ticker",
            tickers,
            index=0 if tickers else None,
            key="ticker_detail_select",
        )

        if selected_ticker:
            row = df[df["Ticker"] == selected_ticker].iloc[0]
            st.metric("Model probability", metric_format(row["Predicted Prob Up (%)"], "%"))
            st.metric("Score", metric_format(row["Score"]))
            st.metric("Rating", row["Rating"])
            st.metric("Accuracy", metric_format(row["Accuracy (%)"], "%"))
            st.metric("Precision", metric_format(row["Precision (%)"], "%"))
            st.metric("F1", metric_format(row["F1 Score (%)"], "%"))
            st.metric("RSI 14", metric_format(row["RSI_14"]))
            st.metric("Volatility", metric_format(row["Volatility"], "%"))
            st.markdown("**Top features**")
            st.write(row.get("Top Features", ""))

            st.markdown("### About the Model")
            st.markdown(
                """
                - Scores are generated using historical price-based features and machine learning.
                - Outputs reflect statistical patterns in past data and are not predictions of future performance.
                - Higher scores indicate stronger model signals based on historical relationships.
                
                This tool is designed for research and screening purposes only.
                """
            )

with tabs[1]:
    st.subheader("Ticker chart")
    chart_tickers = filtered["Ticker"].tolist() if not filtered.empty else df["Ticker"].tolist()
    chart_ticker = st.selectbox(
        "Ticker for chart",
        chart_tickers,
        index=0 if chart_tickers else None,
        key="ticker_chart_select",
    )

    if chart_ticker:
        chart_df = load_ticker_chart(chart_ticker)
        if chart_df.empty:
            st.info("No chart data available for this ticker.")
        else:
            chart_long = chart_df.melt("Date", value_vars=["Close", "MA50"], var_name="Series", value_name="Value")
            chart = alt.Chart(chart_long).mark_line().encode(
                x="Date:T",
                y="Value:Q",
                color="Series:N",
                tooltip=["Date:T", "Series:N", "Value:Q"],
            )
            st.altair_chart(chart, use_container_width=True)

with tabs[2]:
    st.subheader("Historical model signal performance")
    if perf.empty:
        st.info("No performance history yet.")
    else:
        perf_market = perf[perf["market"] == market.lower()].copy()

        p1, p2, p3, p4 = st.columns(4)
        p1.metric("Tracked signals", len(perf_market))
        p2.metric(
            "Signal success rate",
            metric_format(perf_market["hit"].mean() * 100, "%")
            if "hit" in perf_market.columns else "-",
        )
        p3.metric(
            "Avg 5d return",
            metric_format(perf_market["forward_return_5d_pct"].mean(), "%")
            if "forward_return_5d_pct" in perf_market.columns else "-",
        )
        p4.metric(
            "Avg signal score",
            metric_format(perf_market["pick_score"].mean())
            if "pick_score" in perf_market.columns else "-",
        )

        recent_cols = [
            "pick_date",
            "market",
            "Ticker",
            "pick_prob_up",
            "pick_score",
            "start_close",
            "end_close_5d",
            "forward_return_5d_pct",
            "hit",
        ]
        available_recent_cols = [c for c in recent_cols if c in perf_market.columns]
        st.dataframe(
            perf_market.sort_values("pick_date", ascending=False)[available_recent_cols],
            use_container_width=True,
            hide_index=True,
        )

with tabs[3]:
    st.subheader("Weekly summary")
    wk = weekly[weekly["market"] == market.lower()].copy() if not weekly.empty else pd.DataFrame()
    if wk.empty:
        st.info("No weekly summary yet.")
    else:
        st.dataframe(wk.sort_values("week_start", ascending=False), use_container_width=True, hide_index=True)

with tabs[4]:
    st.subheader("Benchmarking")
    b = bench[bench["market"] == market.lower()].copy() if not bench.empty else pd.DataFrame()
    if b.empty:
        st.info("No benchmark summary yet.")
    else:
        st.dataframe(b.sort_values("week_start", ascending=False), use_container_width=True, hide_index=True)

        if "alpha_vs_benchmark_pct" in b.columns:
            chart = alt.Chart(b).mark_line(point=True).encode(
                x="week_start:T",
                y="alpha_vs_benchmark_pct:Q",
                tooltip=["week_start:T", "alpha_vs_benchmark_pct:Q"],
            )
            st.altair_chart(chart, use_container_width=True)

with tabs[5]:
    st.subheader("Weekly output")
    wk = weekly[weekly["market"] == market.lower()].copy() if not weekly.empty else pd.DataFrame()
    if wk.empty:
        st.info("No weekly output yet.")
    else:
        wk = wk.sort_values("week_start", ascending=False)
        latest_week = wk["week_start"].iloc[0]
        latest = wk[wk["week_start"] == latest_week]

        if pd.notna(latest_week):
            st.write(f"Week starting: {latest_week.date()}")
        st.dataframe(latest, use_container_width=True, hide_index=True)

        st.download_button(
            "Download latest weekly output",
            data=latest.to_csv(index=False).encode("utf-8"),
            file_name=f"alpharank_{market.lower()}_weekly_output.csv",
            mime="text/csv",
        )

st.markdown("---")
st.markdown(
    '''
**Disclaimer**

AlphaRank provides quantitative research and data-driven rankings.  
It does not provide financial, investment, or trading advice.

No content in this application should be interpreted as a recommendation
to buy or sell any financial instrument.

Users are solely responsible for their own investment decisions.
'''
)
