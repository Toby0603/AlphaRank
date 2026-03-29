from backend_multi_model_utils import BASE_DIR, run_model

if __name__ == "__main__":
    run_model("lightgbm", BASE_DIR / "results_lightgbm.csv")
