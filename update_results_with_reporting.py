import pandas as pd
import yfinance as yf
from pathlib import Path
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from datetime import datetime, timezone

BASE_DIR = Path(__file__).resolve().parent

DATASETS = {
    "core": {
        "tickers_file": BASE_DIR / "tickers_core.csv",
        "output_file": BASE_DIR / "results_core.csv",
        "max_tickers": 50,
        "benchmark": "^GSPC",
    },
    "ftse": {
        "tickers_file": BASE_DIR / "tickers_ftse.csv",
        "output_file": BASE_DIR / "results_ftse.csv",
        "max_tickers": 500,
        "benchmark": "^FTSE",
    },
    "us": {
        "tickers_file": BASE_DIR / "tickers_us.csv",
        "output_file": BASE_DIR / "results_us.csv",
        "max_tickers": 500,
        "benchmark": "SPY",
    },
    "europe": {
        "tickers_file": BASE_DIR / "tickers_europe.csv",
        "output_file": BASE_DIR / "results_europe.csv",
        "max_tickers": 500,
        "benchmark": "^STOXX50E",
    },
}

PERF_FILE = BASE_DIR / "performance_history.csv"
WEEKLY_FILE = BASE_DIR / "weekly_summary.csv"
BENCH_FILE = BASE_DIR / "benchmark_summary.csv"
FAILED_FILE = BASE_DIR / "failed_tickers.csv"


def compute_rsi(close, window=14):
    close = pd.to_numeric(close, errors="coerce")
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=window, min_periods=window).mean()
    avg_loss = loss.rolling(window=window, min_periods=window).mean()
    rs = avg_gain / avg_loss.replace(0, float("nan"))
    return pd.to_numeric(100 - (100 / (1 + rs)), errors="coerce")


def score_row(row):
    vals = [
        row["Predicted Prob Up (%)"],
        row["Accuracy (%)"],
        row["Precision (%)"],
        row["F1 Score (%)"],
        row["RSI_14"],
    ]
    if any(pd.isna(v) for v in vals):
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


def build_features(df):
    df = df.copy()

    df["Return_1d"] = df["Close"].pct_change()
    df["Return_5d"] = df["Close"].pct_change(5)
    df["Return_10d"] = df["Close"].pct_change(10)
    df["MA50"] = df["Close"].rolling(50).mean()
    df["Momentum_20"] = df["Close"] / df["Close"].shift(20) - 1
    df["Volatility_30"] = df["Return_1d"].rolling(30).std()
    df["Trend_MA50"] = df["Close"] / df["MA50"] - 1

    rm20 = df["Close"].rolling(20).mean()
    rs20 = df["Close"].rolling(20).std()
    df["Z_Score_20"] = (df["Close"] - rm20) / rs20.replace(0, float("nan"))

    df["Volume_MA_Ratio"] = df["Volume"] / df["Volume"].rolling(20).mean()
    df["RSI_14"] = compute_rsi(df["Close"], 14)

    df["forward_return_5d"] = df["Close"].shift(-5) / df["Close"] - 1
    df["Target"] = (df["forward_return_5d"] > 0.015).astype(int)

    return df


def load_tickers(path, max_tickers):
    df = pd.read_csv(path)
    df["Ticker"] = df["Ticker"].astype(str).str.strip().str.upper()
    return [t for t in df["Ticker"].tolist() if t][:max_tickers]


def download_one(ticker, period="4y"):
    data = yf.download(
        ticker,
        period=period,
        auto_adjust=False,
        progress=False,
        threads=False,
    )

    if data.empty:
        return pd.DataFrame()

    data = data.reset_index()

    if isinstance(data.columns, pd.MultiIndex):
        data.columns = [c[0] if c[0] else c[1] for c in data.columns]

    data = data.loc[:, ~pd.Index(data.columns).duplicated()]

    for col in ["Open", "High", "Low", "Close", "Volume"]:
        if col in data.columns:
            if isinstance(data[col], pd.DataFrame):
                data[col] = data[col].iloc[:, 0]
            data[col] = pd.to_numeric(data[col], errors="coerce")

    if "Date" in data.columns:
        data["Date"] = pd.to_datetime(data["Date"], errors="coerce")

    return data


def walk_forward_validate(X, y, feature_cols):
    n = len(X)

    initial_train_size = max(120, int(n * 0.5))
    test_window = max(20, int(n * 0.1))
    step_size = test_window

    if n < initial_train_size + test_window:
        return None

    fold_metrics = []
    last_model = None
    start_test = initial_train_size

    while start_test + test_window <= n:
        X_train = X.iloc[:start_test].copy()
        y_train = y.iloc[:start_test].copy()
        X_test = X.iloc[start_test:start_test + test_window].copy()
        y_test = y.iloc[start_test:start_test + test_window].copy()

        pos_count = (y_train == 1).sum()
        neg_count = (y_train == 0).sum()

        if pos_count == 0 or neg_count == 0:
            start_test += step_size
            continue

        scale_pos_weight = neg_count / pos_count

        model = XGBClassifier(
            n_estimators=120,
            max_depth=3,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_weight=3,
            reg_alpha=0.1,
            reg_lambda=1.0,
            scale_pos_weight=scale_pos_weight,
            random_state=42,
            eval_metric="logloss",
            n_jobs=1,
        )

        model.fit(X_train[feature_cols], y_train)

        test_probs = model.predict_proba(X_test[feature_cols])[:, 1]
        classification_threshold = 0.60
        test_preds = (test_probs >= classification_threshold).astype(int)

        fold_metrics.append(
            {
                "Accuracy (%)": accuracy_score(y_test, test_preds) * 100,
                "Precision (%)": precision_score(y_test, test_preds, zero_division=0) * 100,
                "Recall (%)": recall_score(y_test, test_preds, zero_division=0) * 100,
                "F1 Score (%)": f1_score(y_test, test_preds, zero_division=0) * 100,
            }
        )

        last_model = model
        start_test += step_size

    if not fold_metrics or last_model is None:
        return None

    metrics_df = pd.DataFrame(fold_metrics)
    avg_metrics = metrics_df.mean().to_dict()

    return {
        "model": last_model,
        "metrics": avg_metrics,
    }


def process_ticker(ticker):
    try:
        data = download_one(ticker, "4y")

        if data.empty or any(c not in data.columns for c in ["Date", "Open", "High", "Low", "Close", "Volume"]):
            return {
                "failed": True,
                "ticker": ticker,
                "reason": "No price data found",
            }

        data["Ticker"] = ticker
        data = build_features(data)

        feature_cols = [
            "Return_5d",
            "Return_10d",
            "Momentum_20",
            "Volatility_30",
            "Trend_MA50",
            "Z_Score_20",
            "Volume_MA_Ratio",
            "RSI_14",
        ]

        for col in feature_cols:
            data[col] = pd.to_numeric(data[col], errors="coerce")
        data["Target"] = pd.to_numeric(data["Target"], errors="coerce")

        model_data = data.dropna(subset=feature_cols + ["Target"]).copy()
        if len(model_data) < 120:
            return {
                "failed": True,
                "ticker": ticker,
                "reason": "Not enough cleaned rows",
            }

        X = model_data[feature_cols]
        y = model_data["Target"]

        validation_result = walk_forward_validate(X, y, feature_cols)
        if validation_result is None:
            return {
                "failed": True,
                "ticker": ticker,
                "reason": "Not enough data for walk-forward validation",
            }

        model = validation_result["model"]
        accuracy_pct = float(validation_result["metrics"]["Accuracy (%)"])
        precision_pct = float(validation_result["metrics"]["Precision (%)"])
        recall_pct = float(validation_result["metrics"]["Recall (%)"])
        f1_pct = float(validation_result["metrics"]["F1 Score (%)"])

        latest_features = X.tail(1)
        latest_price = float(model_data["Close"].iloc[-1])
        prob_up = float(model.predict_proba(latest_features)[0][1]) * 100
        latest_return_1d = float(model_data["Return_1d"].iloc[-1])
        latest_volatility_30 = float(model_data["Volatility_30"].iloc[-1])
        latest_rsi = float(model_data["RSI_14"].iloc[-1])

        top_features = ", ".join(
            pd.Series(model.feature_importances_, index=feature_cols)
            .sort_values(ascending=False)
            .head(3)
            .index.tolist()
        )

        score = score_row(
            pd.Series(
                {
                    "Predicted Prob Up (%)": prob_up,
                    "Accuracy (%)": accuracy_pct,
                    "Precision (%)": precision_pct,
                    "F1 Score (%)": f1_pct,
                    "RSI_14": latest_rsi,
                }
            )
        )

        latest_idx = model_data.index[-1]
        end_idx = latest_idx + 5 if (latest_idx + 5) in data.index else None
        end_close_5d = float(data.loc[end_idx, "Close"]) if end_idx is not None else None
        forward_return_5d_pct = (((end_close_5d / latest_price) - 1) * 100) if end_close_5d is not None else None
        hit = 1 if (forward_return_5d_pct is not None and forward_return_5d_pct > 2.0) else 0 if forward_return_5d_pct is not None else None

        return {
            "result": {
                "Ticker": ticker,
                "Latest Price ($)": latest_price,
                "Daily Return": latest_return_1d,
                "Volatility": latest_volatility_30,
                "Predicted Prob Up (%)": prob_up,
                "Accuracy (%)": accuracy_pct,
                "Precision (%)": precision_pct,
                "Recall (%)": recall_pct,
                "F1 Score (%)": f1_pct,
                "RSI_14": latest_rsi,
                "Top Features": top_features,
                "Score": score,
                "Rating": rating_from_score(score),
            },
            "perf": {
                "Ticker": ticker,
                "pick_prob_up": prob_up,
                "pick_score": score,
                "start_close": latest_price,
                "end_close_5d": end_close_5d,
                "forward_return_5d_pct": forward_return_5d_pct,
                "hit": hit,
            },
        }

    except Exception as e:
        print(f"Failed {ticker}: {e}")
        return {
            "failed": True,
            "ticker": ticker,
            "reason": str(e),
        }


def append_rows(path, rows):
    if not rows:
        return

    new_df = pd.DataFrame(rows)

    if path.exists():
        old_df = pd.read_csv(path)
        new_df = pd.concat([old_df, new_df], ignore_index=True)

    new_df.to_csv(path, index=False)


def refresh_dataset(name, cfg):
    tickers = load_tickers(cfg["tickers_file"], cfg["max_tickers"])
    results = []
    perf_rows = []
    failed_tickers = []
    pick_date = datetime.now(timezone.utc).date().isoformat()

    for ticker in tickers:
        print(f"Processing {ticker}...")
        item = process_ticker(ticker)

        if item is None:
            failed_tickers.append(
                {
                    "Ticker": ticker,
                    "Reason": "No data returned",
                }
            )
            continue

        if isinstance(item, dict) and item.get("failed"):
            failed_tickers.append(
                {
                    "Ticker": item["ticker"],
                    "Reason": item["reason"],
                }
            )
            continue

        if not isinstance(item, dict) or "result" not in item:
            failed_tickers.append(
                {
                    "Ticker": ticker,
                    "Reason": "Invalid return structure",
                }
            )
            continue

        results.append(item["result"])

        if item["result"].get("Rating") == "Top Ranked":
            perf = item["perf"]
            perf["market"] = name
            perf["pick_date"] = pick_date
            perf["rating"] = item["result"].get("Rating")
            perf_rows.append(perf)

    if results:
        results_df = pd.DataFrame(results)
        results_df.to_csv(cfg["output_file"], index=False)
        print(f"Saved {len(results)} rows to {cfg['output_file']}")
    else:
        print(f"No valid results for {name}")

    if failed_tickers:
        failed_df = pd.DataFrame(failed_tickers)

        if FAILED_FILE.exists():
            try:
                old_failed = pd.read_csv(FAILED_FILE)
                if not old_failed.empty:
                    failed_df = pd.concat([old_failed, failed_df], ignore_index=True)
            except pd.errors.EmptyDataError:
                pass

        failed_df.drop_duplicates(subset=["Ticker"], inplace=True)
        failed_df.to_csv(FAILED_FILE, index=False)
        print(f"Saved {len(failed_df)} failed tickers to {FAILED_FILE}")

    return perf_rows


def build_weekly_summary():
    if not PERF_FILE.exists():
        return

    perf = pd.read_csv(PERF_FILE)
    if perf.empty:
        return

    perf["pick_date"] = pd.to_datetime(perf["pick_date"], errors="coerce")
    perf["week_start"] = perf["pick_date"].dt.to_period("W").apply(lambda r: r.start_time)

    summary = (
        perf.groupby(["market", "week_start"], dropna=True)
        .agg(
            picks=("Ticker", "count"),
            hit_rate=("hit", "mean"),
            avg_return_5d_pct=("forward_return_5d_pct", "mean"),
            avg_pick_prob_up=("pick_prob_up", "mean"),
            avg_pick_score=("pick_score", "mean"),
        )
        .reset_index()
    )

    summary["hit_rate"] = summary["hit_rate"] * 100
    summary.to_csv(WEEKLY_FILE, index=False)


def build_benchmark_summary():
    if not WEEKLY_FILE.exists():
        return

    weekly = pd.read_csv(WEEKLY_FILE)
    if weekly.empty:
        return

    weekly["week_start"] = pd.to_datetime(weekly["week_start"], errors="coerce")
    rows = []

    for market, cfg in DATASETS.items():
        bench = download_one(cfg["benchmark"], "2y")
        if bench.empty or "Date" not in bench.columns or "Close" not in bench.columns:
            continue

        bench = bench.sort_values("Date").reset_index(drop=True)
        bench["forward_return_5d_pct"] = (bench["Close"].shift(-5) / bench["Close"] - 1) * 100
        bench["week_start"] = bench["Date"].dt.to_period("W").apply(lambda r: r.start_time)

        bench_weekly = (
            bench.groupby("week_start", dropna=True)
            .agg(benchmark_return_5d_pct=("forward_return_5d_pct", "mean"))
            .reset_index()
        )

        merged = weekly[weekly["market"] == market].merge(bench_weekly, on="week_start", how="left")
        merged["alpha_vs_benchmark_pct"] = merged["avg_return_5d_pct"] - merged["benchmark_return_5d_pct"]
        merged["benchmark_ticker"] = cfg["benchmark"]
        rows.append(merged)

    if rows:
        pd.concat(rows, ignore_index=True).to_csv(BENCH_FILE, index=False)


def main():
    all_perf = []

    for name, cfg in DATASETS.items():
        all_perf.extend(refresh_dataset(name, cfg))

    append_rows(PERF_FILE, all_perf)
    build_weekly_summary()
    build_benchmark_summary()

    print("Updated results, performance history, weekly summary, and benchmark summary")


if __name__ == "__main__":
    main()
