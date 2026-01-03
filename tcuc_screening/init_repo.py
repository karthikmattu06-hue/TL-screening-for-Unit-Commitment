from pathlib import Path

structure = {
    "configs": ["rts96.yaml", "experiment.yaml"],
    "data_raw/oasys_ieee96": [],
    "data_processed/rts96_v1": [
        "flows.parquet", "loadings.parquet", "demands.parquet", "metadata.json"
    ],
    "datasets/rts96_v1_D8": [
        "XY_train.pt", "XY_test.pt", "scalers.pkl", "index_splits.json"
    ],
    "models": ["lstm_regressor.py", "train.py", "infer.py"],
    "screening": ["thresholding.py", "evaluate_accuracy.py", "build_active_sets.py"],
    "uc_eval": ["solve_full.py", "solve_screened.py", "constraint_generation.py", "metrics.py"],
    "notebooks": [],
    "scripts": [
        "01_fetch_raw_data.py",
        "02_generate_labels.py",
        "03_build_windows.py",
        "04_train_model.py",
        "05_screen_and_eval_uc.py",
    ],
}

root = Path("tcuc_screening")

for folder, files in structure.items():
    dir_path = root / folder
    dir_path.mkdir(parents=True, exist_ok=True)
    for f in files:
        (dir_path / f).touch()

print("Repository structure created.")
