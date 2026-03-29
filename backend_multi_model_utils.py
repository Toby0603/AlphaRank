
import pandas as pd
import yfinance as yf
from pathlib import Path
from datetime import datetime, timezone
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier

BASE_DIR = Path(__file__).resolve().parent

MARKETS = {
    "ftse": {"tickers_file": BASE_DIR / "tickers_ftse.csv", "max_tickers": 300},
    "us": {"tickers_file": BASE_DIR / "tickers_us.csv", "max_tickers": 300},
    "europe": {"tickers_file": BASE_DIR / "tickers_europe.csv", "max_tickers": 300},
}

PERF_FILE = BASE_DIR / "performance_history.csv"
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
    vals = [row["Predicted Prob Up (%)"], row["Accuracy (%)"], row["Precision (%)"], row["F1 Score (%)"], row["RSI_14"]]
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

def load_all_tickers():
    all_tickers = []
    for _, cfg in MARKETS.items():
        path = cfg["tickers_file"]
        if not path.exists():
            continue
        df = pd.read_csv(path)
        df["Ticker"] = df["Ticker"].astype(str).str.strip().str.upper()
        all_tickers.extend([t for t in df["Ticker"].tolist() if t][:cfg["max_tickers"]])
    return list(dict.fromkeys(all_tickers))

def download_one(ticker, period="4y"):
    data = yf.download(ticker, period=period, auto_adjust=False, progress=False, threads=False)
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

def make_model(model_name, scale_pos_weight):
    model_name = model_name.lower()
    if model_name == "xgboost":
        return XGBClassifier(
            n_estimators=120, max_depth=3, learning_rate=0.05, subsample=0.8,
            colsample_bytree=0.8, min_child_weight=3, reg_alpha=0.1, reg_lambda=1.0,
            scale_pos_weight=scale_pos_weight, random_state=42, eval_metric="logloss", n_jobs=1,
        )
    if model_name == "catboost":
        return CatBoostClassifier(
            iterations=200, depth=5, learning_rate=0.05, loss_function="Logloss",
            eval_metric="Logloss", random_seed=42, verbose=0, scale_pos_weight=scale_pos_weight,
        )
    if model_name == "lightgbm":
        return LGBMClassifier(
            n_estimators=200, max_depth=5, learning_rate=0.05, subsample=0.8,
            colsample_bytree=0.8, min_child_samples=20, reg_alpha=0.1, reg_lambda=1.0,
            scale_pos_weight=scale_pos_weight, random_state=42, n_jobs=1, verbosity=-1,
        )
    raise ValueError(f"Unsupported model: {model_name}")

def walk_forward_validate(X, y, feature_cols, model_name):
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
        model = make_model(model_name, scale_pos_weight)
        model.fit(X_train[feature_cols], y_train)
        test_probs = model.predict_proba(X_test[feature_cols])[:, 1]
        test_preds = (test_probs >= 0.60).astype(int)
        fold_metrics.append({
            "Accuracy (%)": accuracy_score(y_test, test_preds) * 100,
            "Precision (%)": precision_score(y_test, test_preds, zero_division=0) * 100,
            "Recall (%)": recall_score(y_test, test_preds, zero_division=0) * 100,
            "F1 Score (%)": f1_score(y_test, test_preds, zero_division=0) * 100,
        })
        last_model = model
        start_test += step_size
    if not fold_metrics or last_model is None:
        return None
    metrics_df = pd.DataFrame(fold_metrics)
    return {"model": last_model, "metrics": metrics_df.mean().to_dict()}

def process_ticker(ticker, model_name):
    try:
        data = download_one(ticker, "4y")
        if data.empty or any(c not in data.columns for c in ["Date","Open","High","Low","Close","Volume"]):
            return {"failed": True, "ticker": ticker, "reason": "No price data found"}
        data["Ticker"] = ticker
        data = build_features(data)
        feature_cols = ["Return_5d","Return_10d","Momentum_20","Volatility_30","Trend_MA50","Z_Score_20","Volume_MA_Ratio","RSI_14"]
        for col in feature_cols:
            data[col] = pd.to_numeric(data[col], errors="coerce")
        data["Target"] = pd.to_numeric(data["Target"], errors="coerce")
        model_data = data.dropna(subset=feature_cols + ["Target"]).copy()
        if len(model_data) < 120:
            return {"failed": True, "ticker": ticker, "reason": "Not enough cleaned rows"}
        X = model_data[feature_cols]
        y = model_data["Target"]
        validation_result = walk_forward_validate(X, y, feature_cols, model_name)
        if validation_result is None:
            return {"failed": True, "ticker": ticker, "reason": "Not enough data for walk-forward validation"}
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
        if hasattr(model, "feature_importances_"):
            importances = pd.Series(model.feature_importances_, index=feature_cols)
            top_features = ", ".join(importances.sort_values(ascending=False).head(3).index.tolist())
        else:
            top_features = ", ".join(feature_cols[:3])
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
                "Model": model_name,
            },
            "perf": {
                "Ticker": ticker,
                "pick_prob_up": prob_up,
                "pick_score": score,
                "start_close": latest_price,
                "end_close_5d": end_close_5d,
                "forward_return_5d_pct": forward_return_5d_pct,
                "hit": hit,
                "model": model_name,
            },
        }
    except Exception as e:
        return {"failed": True, "ticker": ticker, "reason": str(e)}

def append_performance_rows(rows):
    if not rows:
        return
    new_df = pd.DataFrame(rows)
    if PERF_FILE.exists():
        try:
            old_df = pd.read_csv(PERF_FILE)
            combined = pd.concat([old_df, new_df], ignore_index=True)
        except pd.errors.EmptyDataError:
            combined = new_df.copy()
    else:
        combined = new_df.copy()
    key_cols = ["pick_date", "Ticker", "start_close", "model"]
    if all(col in combined.columns for col in key_cols):
        combined["pick_date"] = combined["pick_date"].astype(str)
        combined["Ticker"] = combined["Ticker"].astype(str).str.strip().str.upper()
        combined["start_close"] = pd.to_numeric(combined["start_close"], errors="coerce").round(6)
        combined["model"] = combined["model"].astype(str).str.strip().str.lower()
        combined["_has_end_close"] = combined["end_close_5d"].notna().astype(int) if "end_close_5d" in combined.columns else 0
        combined["_has_forward_ret"] = combined["forward_return_5d_pct"].notna().astype(int) if "forward_return_5d_pct" in combined.columns else 0
        combined["_has_hit"] = combined["hit"].notna().astype(int) if "hit" in combined.columns else 0
        combined["_row_order"] = range(len(combined))
        combined = combined.sort_values(by=key_cols + ["_has_end_close","_has_forward_ret","_has_hit","_row_order"], ascending=True)
        combined = combined.drop_duplicates(subset=key_cols, keep="last")
        combined = combined.drop(columns=["_has_end_close","_has_forward_ret","_has_hit","_row_order"], errors="ignore")
    combined.to_csv(PERF_FILE, index=False)

def append_failed_rows(rows):
    if not rows:
        return
    new_df = pd.DataFrame(rows)
    if FAILED_FILE.exists():
        try:
            old_df = pd.read_csv(FAILED_FILE)
            combined = pd.concat([old_df, new_df], ignore_index=True)
        except pd.errors.EmptyDataError:
            combined = new_df.copy()
    else:
        combined = new_df.copy()
    dedupe_cols = [c for c in ["Ticker", "model", "Reason"] if c in combined.columns]
    if dedupe_cols:
        combined = combined.drop_duplicates(subset=dedupe_cols, keep="last")
    combined.to_csv(FAILED_FILE, index=False)

def run_model(model_name, output_file):
    tickers = load_all_tickers()
    results, perf_rows, failed_rows = [], [], []
    pick_date = datetime.now(timezone.utc).date().isoformat()
    print(f"Running {model_name} across {len(tickers)} tickers")
    for ticker in tickers:
        print(f"Processing {ticker} [{model_name}]...")
        item = process_ticker(ticker, model_name)
        if item is None:
            failed_rows.append({"Ticker": ticker, "Reason": "No data returned", "model": model_name})
            continue
        if isinstance(item, dict) and item.get("failed"):
            failed_rows.append({"Ticker": item["ticker"], "Reason": item["reason"], "model": model_name})
            continue
        if not isinstance(item, dict) or "result" not in item:
            failed_rows.append({"Ticker": ticker, "Reason": "Invalid return structure", "model": model_name})
            continue
        results.append(item["result"])
        if item["result"].get("Rating") == "Top Ranked":
            perf = item["perf"]
            perf["pick_date"] = pick_date
            perf_rows.append(perf)
    if results:
        pd.DataFrame(results).sort_values(["Score", "Predicted Prob Up (%)"], ascending=False).to_csv(output_file, index=False)
        print(f"Saved {len(results)} rows to {output_file}")
    else:
        print(f"No valid results for {model_name}")
    append_performance_rows(perf_rows)
    append_failed_rows(failed_rows)
    print(f"Appended {len(perf_rows)} performance rows for {model_name}")
    print(f"Appended {len(failed_rows)} failed rows for {model_name}")
