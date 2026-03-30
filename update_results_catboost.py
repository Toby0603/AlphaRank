from backend_multi_model_utils import BASE_DIR, run_model

if __name__ == "__main__":
    run_varinat("catboost", "base" BASE_DIR / "results_catboost_base.csv")
    run_varinat("catboost", "interactions" BASE_DIR / "results_catboost_interactions.csv")
