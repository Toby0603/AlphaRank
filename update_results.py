import pandas as pd
import yfinance as yf
from pathlib import Path
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

BASE_DIR = Path(__file__).resolve().parent
TICKERS_FILE = BASE_DIR / "tickers.csv"
OUTPUT_FILE = BASE_DIR / "results.csv"

REQUIRED_TICKER_COLUMNS = ["Ticker"]

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

def build_features(data: pd.DataFrame) -> pd.DataFrame:
    df = data.copy()
    df["Return_1d"] = df["Close"].pct_change()
    df["Return_5d"] = df["Close"].pct_change(5)
    df["Return_10d"] = df["Close"].pct_change(10)
    df["MA20"] = df["Close"].rolling(20).mean()
    df["MA50"] = df["Close"].rolling(50).mean()
    df["MA200"] = df["Close"].rolling(200).mean()
    df["Momentum_5"] = df["Close"] / df["Close"].shift(5) - 1
    df["Momentum_20"] = df["Close"] / df["Close"].shift(20) - 1
    df["Volatility_10"] = df["Return_1d"].rolling(10).std()
    df["Volatility_30"] = df["Return_1d"].rolling(30).std()
    df["Trend_MA20"] = df["Close"] / df["MA20"] - 1
    df["Trend_MA50"] = df["Close"] / df["MA50"] - 1
    rolling_mean_20 = df["Close"].rolling(20).mean()
    rolling_std_20 = df["Close"].rolling(20).std()
    df["Z_Score_20"] = (df["Close"] - rolling_mean_20) / rolling_std_20.replace(0, float("nan"))
    df["Volume_Change"] = df["Volume"].pct_change()
    df["Volume_MA_Ratio"] = df["Volume"] / df["Volume"].rolling(20).mean()
    df["RSI_14"] = compute_rsi(df["Close"], 14)
    df["Return_Lag1"] = df["Return_1d"].shift(1)
    df["Return_Lag2"] = df["Return_1d"].shift(2)
    df["Return_Lag3"] = df["Return_1d"].shift(3)

    future_return_5d = df["Close"].shift(-5) / df["Close"] - 1
    df["Target"] = (future_return_5d > 0.02).astype(int)
    return df

def load_tickers() -> pd.DataFrame:
    if not TICKERS_FILE.exists():
        raise FileNotFoundError(f"Missing tickers file: {TICKERS_FILE}")

    df = pd.read_csv(TICKERS_FILE)
    missing = [c for c in REQUIRED_TICKER_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"tickers.csv is missing columns: {', '.join(missing)}")

    df["Ticker"] = df["Ticker"].astype(str).str.strip().str.upper()
    df = df[df["Ticker"] != ""].drop_duplicates(subset=["Ticker"]).reset_index(drop=True)
    return df

def process_ticker(ticker: str):
    data = yf.download(ticker, period="2y", auto_adjust=False, progress=False)
    if data.empty:
        return None

    data = data.reset_index()
    data.columns = [c[0] if isinstance(c, tuple) else c for c in data.columns]

    required = {"Date", "Open", "High", "Low", "Close", "Volume"}
    if not required.issubset(data.columns):
        return None

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
        return None

    X = model_data[feature_cols]
    y = model_data["Target"]

    split_idx = int(len(model_data) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    if len(X_train) < 80 or len(X_test) < 20:
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

    feature_importance = pd.Series(model.feature_importances_, index=feature_cols).sort_values(ascending=False)
    top_features = ", ".join(feature_importance.head(3).index.tolist())

    return {
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
    }

def main():
    tickers_df = load_tickers()
    results = []
    skipped = []

    for ticker in tickers_df["Ticker"]:
        print(f"Processing {ticker}...")
        row = process_ticker(ticker)
        if row is None:
            skipped.append(ticker)
        else:
            results.append(row)

    if not results:
        raise RuntimeError("No results were generated. Check your tickers list and data source.")

    df = pd.DataFrame(results).sort_values("Predicted Prob Up (%)", ascending=False)
    df.to_csv(OUTPUT_FILE, index=False)

    print(f"Saved {len(df)} rows to {OUTPUT_FILE}")
    if skipped:
        print("Skipped tickers:", ", ".join(skipped))

if __name__ == "__main__":
    main()
