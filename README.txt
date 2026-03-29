AlphaRank multi-model backend pack

Files:
- backend_multi_model_utils.py
- update_results_xgboost.py
- update_results_catboost.py
- update_results_lightgbm.py
- .github/workflows/refresh-all-models.yml
- requirements_backend.txt

Outputs:
- results_xgboost.csv
- results_catboost.csv
- results_lightgbm.csv
- performance_history.csv with model column
- failed_tickers.csv with model column

Universe:
- FTSE
- US
- Europe
(No Core)
