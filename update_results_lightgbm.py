from backend_multi_model_utils import BASE_DIR, run_variant

if __name__ == "__main__":
    run_variant("lightgbm", "base", BASE_DIR / "results_lightgbm_base.csv")
    run_variant("lightgbm", "interactions", BASE_DIR / "results_lightgbm_interactions.csv")
