# AlphaRank Final App Version

This is a product-style Streamlit dashboard built for speed and cleaner monetisation.

## What changed
- No live model training in the app
- No BUY/SELL signal column
- Uses a precomputed `results.csv`
- Includes login with `APP_USERNAME` and `APP_PASSWORD`
- Adds filters, ranking, download, and ticker detail view

## Files
- `app.py`
- `requirements.txt`
- `results.csv` (sample data)
- `README.md`

## Environment variables
Set these on your host:
- `APP_USERNAME`
- `APP_PASSWORD`

## Run locally
```bash
pip install -r requirements.txt
export APP_USERNAME=demo
export APP_PASSWORD=demo123
streamlit run app.py
```

## Expected CSV columns
- Ticker
- Latest Price ($)
- Daily Return
- Volatility
- Predicted Prob Up (%)
- Accuracy (%)
- Precision (%)
- Recall (%)
- F1 Score (%)
- RSI_14
- Top Features

Optional columns like `Score` and `Rating` can be included, but the app will create them if missing.

## Product direction
This version is built for a fast hosted dashboard. The model pipeline should run separately offline and write updated results into `results.csv`.
