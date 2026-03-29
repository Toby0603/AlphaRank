from backend_multi_model_utils import BASE_DIR, run_model

if __name__ == "__main__":
    run_model("catboost", BASE_DIR / "results_catboost.csv")
