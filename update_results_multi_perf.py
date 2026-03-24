import pandas as pd
import yfinance as yf
from pathlib import Path
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone

BASE_DIR = Path(__file__).resolve().parent

DATASETS = {
    "core": {"tickers_file": BASE_DIR / "tickers_core.csv", "output_file": BASE_DIR / "results_core.csv", "max_tickers": 100},
    "ftse": {"tickers_file": BASE_DIR / "tickers_ftse.csv", "output_file": BASE_DIR / "results_ftse.csv", "max_tickers": 250},
    "us": {"tickers_file": BASE_DIR / "tickers_us.csv", "output_file": BASE_DIR / "results_us.csv", "max_tickers": 500},
}

PERF_FILE = BASE_DIR / "performance_history.csv"

def compute_rsi(close: pd.Series, window: int = 14) -> pd.Series:
    close = pd.to_numeric(close, errors="coerce")
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=window, min_periods=window).mean()
    avg_loss = loss.rolling(window=window, min_periods=window).mean()
    rs = avg_gain / avg_loss.replace(0, float("nan"))
    rsi = 100 - (100 / (1 + rs))
    return pd.to_numeric(rsi, errors="coerce")

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

def build_features(data: pd.DataFrame) -> pd.DataFrame:
    df = data.copy()
    df["Return_1d"] = df["Close"].pct_change()
    df["Return_5d"] = df["Close"].pct_change(5)
    df["Return_10d"] = df["Close"].pct_change(10)
    df["MA20"] = df["Close"].rolling(20).mean()
    df["MA50"] = df["Close"].rolling(50).mean()
    df["MA200"] = df["Close"].rolling(200).mean()
    df["Momentum_20"] = df["Close"] / df["Close"].shift(20) - 1
    df["Volatility_30"] = df["Return_1d"].rolling(30).std()
    df["Trend_MA50"] = df["Close"] / df["MA50"] - 1
    rolling_mean_20 = df["Close"].rolling(20).mean()
    rolling_std_20 = df["Close"].rolling(20).std()
    df["Z_Score_20"] = (df["Close"] - rolling_mean_20) / rolling_std_20.replace(0, float("nan"))
    df["Volume_MA_Ratio"] = df["Volume"] / df["Volume"].rolling(20).mean()
    df["RSI_14"] = compute_rsi(df["Close"], 14)
    future_return_5d = df["Close"].shift(-5) / df["Close"] - 1
    df["Target"] = (future_return_5d > 0.02).astype(int)
    return df

def load_tickers(path: Path, max_tickers: int):
    df = pd.read_csv(path)
    df["Ticker"] = df["Ticker"].astype(str).str.strip().str.upper()
    return [t for t in df["Ticker"].tolist() if t][:max_tickers]

def process_ticker(ticker: str):
    try:
        data = yf.download(
            ticker,
            period="2y",
            auto_adjust=False,
            progress=False,
            threads=False,
        )

        if data.empty:
            print(f"Skipped {ticker}: empty download")
            return None

        data = data.reset_index()

        # Flatten yfinance columns safely
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = [c[0] if c[0] else c[1] for c in data.columns]

        # Drop duplicate column names if Yahoo returns odd structures
        data = data.loc[:, ~pd.Index(data.columns).duplicated()]

        required = ["Date", "Open", "High", "Low", "Close", "Volume"]
        missing = [col for col in required if col not in data.columns]
        if missing:
            print(f"Skipped {ticker}: missing columns {missing}")
            return None

        # Force OHLCV columns to single numeric series
        for col in ["Open", "High", "Low", "Close", "Volume"]:
            if isinstance(data[col], pd.DataFrame):
                data[col] = data[col].iloc[:, 0]
            data[col] = pd.to_numeric(data[col], errors="coerce")

        data["Date"] = pd.to_datetime(data["Date"], errors="coerce")
        data["Ticker"] = ticker

        if data["Close"].isna().all():
            print(f"Skipped {ticker}: invalid Close data")
            return None

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
            print(f"Skipped {ticker}: insufficient model rows ({len(model_data)})")
            return None

        X = model_data[feature_cols]
        y = model_data["Target"]

        split_idx = int(len(model_data) * 0.8)
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

        if len(X_train) < 80 or len(X_test) < 20:
            print(f"Skipped {ticker}: insufficient train/test split")
            return None

        model = XGBClassifier(
            n_estimators=120,
            max_depth=3,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_weight=3,
            reg_alpha=0.1,
            reg_lambda=1.0,
            random_state=42,
            eval_metric="logloss",
            n_jobs=1,
        )
        model.fit(X_train, y_train)

        test_preds = model.predict(X_test)
        accuracy_pct = accuracy_score(y_test, test_preds) * 100
        precision_pct = precision_score(y_test, test_preds, zero_division=0) * 100
        recall_pct = recall_score(y_test, test_preds, zero_division=0) * 100
        f1_pct = f1_score(y_test, test_preds, zero_division=0) * 100

        latest_features = X.tail(1)
        latest_price = float(model_data["Close"].iloc[-1])
        prob_up = float(model.predict_proba(latest_features)[0][1]) * 100
        latest_return_1d = float(model_data["Return_1d"].iloc[-1])
        latest_volatility_30 = float(model_data["Volatility_30"].iloc[-1])
        latest_rsi = float(model_data["RSI_14"].iloc[-1])

        feature_importance = pd.Series(
            model.feature_importances_, index=feature_cols
        ).sort_values(ascending=False)
        top_features = ", ".join(feature_importance.head(3).index.tolist())

        score = score_row(pd.Series({
            "Predicted Prob Up (%)": prob_up,
            "Accuracy (%)": accuracy_pct,
            "Precision (%)": precision_pct,
            "F1 Score (%)": f1_pct,
            "RSI_14": latest_rsi,
        }))

        latest_idx = model_data.index[-1]
        end_idx = latest_idx + 5 if (latest_idx + 5) in data.index else None
        end_close_5d = float(data.loc[end_idx, "Close"]) if end_idx is not None else None
        forward_return_5d_pct = (
            ((end_close_5d / latest_price) - 1) * 100
            if end_close_5d is not None else None
        )
        hit = (
            1 if (forward_return_5d_pct is not None and forward_return_5d_pct > 2.0)
            else 0 if forward_return_5d_pct is not None
            else None
        )

        print(
            f"{ticker}: price={latest_price:.2f}, "
            f"prob_up={prob_up:.1f}, "
            f"score={score}, "
            f"rsi={latest_rsi:.1f}, "
            f"vol={latest_volatility_30:.4f}"
        )

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
        return None

def append_performance_rows(rows):
    if not rows:
        return

    new_df = pd.DataFrame(rows)
    if PERF_FILE.exists():
        old_df = pd.read_csv(PERF_FILE)
        combined = pd.concat([old_df, new_df], ignore_index=True)
    else:
        combined = new_df

    combined.to_csv(PERF_FILE, index=False)

def refresh_dataset(name: str, cfg: dict):
    tickers = load_tickers(cfg["tickers_file"], cfg["max_tickers"])
    print(f"Refreshing {name} with {len(tickers)} tickers")

    results = []
    perf_rows = []
    pick_date = datetime.now(timezone.utc).date().isoformat()

    for ticker in tickers:
        print(f"Processing {ticker}...")
        item = process_ticker(ticker)
        if item is not None:
            results.append(item["result"])
            perf = item["perf"]
            perf["market"] = name
            perf["pick_date"] = pick_date
            perf_rows.append(perf)

    if not results:
        raise RuntimeError(f"No results were generated for {name}")

    df = pd.DataFrame(results)
    dupe_check = df[["Latest Price ($)", "Predicted Prob Up (%)", "RSI_14"]].duplicated().sum()
    print(f"{name}: duplicated metric rows = {dupe_check}")

    df = df.sort_values(["Score", "Predicted Prob Up (%)"], ascending=False)
    df.to_csv(cfg["output_file"], index=False)
    print(f"Saved {len(df)} rows to {cfg['output_file']}")
    return perf_rows

def main():
    all_perf = []
    for name, cfg in DATASETS.items():
        all_perf.extend(refresh_dataset(name, cfg))
    append_performance_rows(all_perf)
    print(f"Updated performance history: {PERF_FILE}")

if __name__ == "__main__":
    main()
