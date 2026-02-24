import argparse
import numpy as np
import pandas as pd
import npeet.entropy_estimators as ee
import os
import json
import traceback
from tableshift import get_dataset

# Configuration
RESULTS_DIR = "results"
OUTPUT_FILE = os.path.join(RESULTS_DIR, "data_shift_summary20k.txt")

os.makedirs(RESULTS_DIR, exist_ok=True)


def append_to_file(text):
    with open(OUTPUT_FILE, "a") as f:
        f.write(text + "\n")


def subsample(X, y, n_samples=20000, seed=0):
    np.random.seed(seed)
    idx = np.random.choice(len(y), min(n_samples, len(y)), replace=False)
    return X[idx], y[idx]


def compute_data_shifts(X_train, y_train, X_test, y_test, seed=0):
    """
    Compute shift measures using NPEET on continuous covariates:
      - Label shift: I(Y; D)
      - Covariate shift: I(X; D)
      - Concept shift: I(Y; D | X)
      - Label-conditioned covariate shift: I(X; D | Y)
    where D is a binary domain indicator (0=train, 1=test).
    """
    # Subsample to ~5000 points for speed
    X_train_s, y_train_s = subsample(X_train, y_train, seed=seed)
    X_test_s, y_test_s = subsample(X_test, y_test, seed=seed + 1)

    # Construct domain indicator D
    n_train = len(y_train_s)
    n_test = len(y_test_s)
    D = np.concatenate([np.zeros(n_train), np.ones(n_test)]).reshape(-1, 1)

    # Combine data
    Y = np.concatenate([y_train_s, y_test_s]).reshape(-1, 1)
    X = np.vstack([X_train_s, X_test_s])

    # Estimate shifts
    label_shift = np.abs(np.mean(y_train_s) - np.mean(y_test_s))
    cov_shift = ee.mi(X, D)
    concept = ee.cmi(Y, D, X)
    lcov = ee.cmi(X, D, Y)

    return label_shift, cov_shift, concept, lcov


def process_experiment(dataset_name, cache_dir, n_runs=10):
    """
    Run the shift computation n_runs times and record mean and std.
    """
    try:
        ds = get_dataset(dataset_name, cache_dir)
        X_train_df, y_train_df, _, _ = ds.get_pandas("train")
        X_test_df, y_test_df, _, _ = ds.get_pandas("ood_test")
    except Exception:
        print(f"Failed loading dataset {dataset_name}")
        return

    # Convert to numpy
    X_train = X_train_df.values.astype(np.float32)
    y_train = y_train_df.values
    X_test = X_test_df.values.astype(np.float32)
    y_test = y_test_df.values

    # Collect results
    results = np.zeros((n_runs, 4), dtype=float)
    for i in range(n_runs):
        try:
            results[i] = compute_data_shifts(X_train, y_train, X_test, y_test, seed=i)
        except Exception as e:
            print(f"Run {i} failed for {dataset_name}: {e}")
            results[i] = np.nan

    # Compute statistics
    means = np.nanmean(results, axis=0)
    stds = np.nanstd(results, axis=0)

    # Write summary text
    summary = [f"Dataset={dataset_name}"]
    names = ["Label Shift", "Covariate Shift", "Concept Shift", "Label-Cond Cov Shift"]
    for j, name in enumerate(names):
        summary.append(f"  {name:25s} mean={means[j]:.4f}, std={stds[j]:.4f}")
    summary.append("----------------------------------------")
    append_to_file("\n".join(summary))

    # Save JSON
    json_path = os.path.join(RESULTS_DIR, f"data_shifts_{dataset_name}.json")
    json_data = {names[j]: {"mean": float(means[j]), "std": float(stds[j])} for j in range(4)}
    with open(json_path, 'w') as f:
        json.dump(json_data, f, indent=4)

    print(f"Finished {dataset_name}: means={means}, stds={stds}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cache_dir', default='tmp')
    parser.add_argument('--runs', type=int, default=10, help='Number of repetitions')
    args = parser.parse_args()

    datasets = [
        "diabetes_readmission",
        "acsfoodstamps",
        "acsincome",
        "acspubcov",
        "acsunemployment",
        "brfss_diabetes",
        "brfss_blood_pressure",
        "assistments",
    ]

    for ds_name in datasets:
        print(f"Processing {ds_name} over {args.runs} runs")
        try:
            process_experiment(ds_name, args.cache_dir, n_runs=args.runs)
        except Exception as e:
            print(f"Error in {ds_name}: {e}")
            traceback.print_exc()
