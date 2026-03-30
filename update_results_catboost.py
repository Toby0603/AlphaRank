from backend_multi_model_utils import BASE_DIR, run_variant

if __name__ == "__main__":
    run_variant("catboost", "base", BASE_DIR / "results_catboost_base.csv")
    run_variant("catboost", "interactions", BASE_DIR / "results_catboost_interactions.csv")
