from backend_multi_model_utils import BASE_DIR, run_variant

if __name__ == "__main__":
    run_variant("xgboost", "base", BASE_DIR / "results_xgboost_base.csv")
    run_variant("xgboost", "interactions", BASE_DIR / "results_xgboost_interactions.csv")
