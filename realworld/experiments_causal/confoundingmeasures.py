import argparse
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mutual_info_score
from sklearn.preprocessing import KBinsDiscretizer
import traceback

from tableshift import get_dataset
from tableshift.models.training import train
from tableshift.models.utils import get_estimator
from tableshift.models.default_hparams import get_default_config

# Define device (ensure consistency)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
def ensure_2d(arr):
    """
    Ensures that the input array is 2D by adding a dummy dimension if necessary.
    """
    arr = np.array(arr)
    if arr.ndim == 1:
        return arr[:, np.newaxis]  # Add a dummy dimension at the end
    return arr

def flatten_2d_to_1d(arr):
    """
    Flattens a 2D array into a 1D array by treating each row as a unique identifier.
    """
    return np.array([hash(tuple(row)) for row in arr])

def compute_conditional_mutual_information(x, y, z):
    """
    Compute conditional mutual information I(X; Y | Z) using the formula:
    I(X; Y | Z) = I(X; Y, Z) - I(X; Z)
    """
    # Ensure y and z are 2D
    y = ensure_2d(np.array(y))
    z = ensure_2d(np.array(z))

    # Stack y and z to compute I(X; Y, Z)
    yz = np.hstack([y, z])

    # Flatten yz and z into 1D arrays
    yz_flat = flatten_2d_to_1d(yz)
    z_flat = flatten_2d_to_1d(z)

    # Compute I(X; Y, Z)
    mi_xy_z = mutual_info_score(x, yz_flat)

    # Compute I(X; Z)
    mi_x_z = mutual_info_score(x, z_flat)

    # CMI = I(X; Y, Z) - I(X; Z)
    return mi_xy_z - mi_x_z

def compute_conditional_confounding(data_features, labels, domains, informative, spurious, n_bins=5):
    """
    Computes the average conditional mutual information between the label and each spurious feature,
    conditioned on the informative features, across domains.
    """
    df = data_features.copy()
    df['label'] = labels
    df['domain'] = domains

    domain_means = df.groupby('domain').mean()

    # Remove constant features
    domain_means = domain_means.loc[:, (domain_means != domain_means.iloc[0]).any()]
    
    discretized = domain_means.copy()
    discretizer = KBinsDiscretizer(n_bins=n_bins, encode='ordinal', strategy='uniform')
    for col in discretized.columns:
        col_data = discretized[col].values.reshape(-1, 1)
        discretized[col] = discretizer.fit_transform(col_data).astype(int).ravel()

    cmi = None
    label_arr = discretized['label'].tolist()
    spurious_arr = discretized[np.intersect1d(discretized.columns, spurious)].values.tolist()

    cond_arr = discretized[np.intersect1d(discretized.columns, informative)].values.tolist() if informative else None
    
    if cond_arr is not None:
        # Compute CMI using the formula I(X; Y | Z) = I(X; Y, Z) - I(X; Z)
        cmi = compute_conditional_mutual_information(label_arr, spurious_arr, cond_arr)
    else:
        # If no conditioning features, compute mutual information directly
        cmi = mutual_info_score(label_arr, spurious_arr)
    
    return cmi

def main(experiment, cache_dir, model, debug: bool):
    """
    Main function to run the experiment.
    """
    if debug:
        print("[INFO] Running in debug mode.")
        experiment = {k: v + "_debug" for k, v in experiment.items()}

    causal_dataset = get_dataset(experiment['causal'], cache_dir)
    arguablycausal_dataset = get_dataset(experiment['arguablycausal'], cache_dir)
    all_dataset = get_dataset(experiment['all'], cache_dir)

    X_c, _, _, _ = causal_dataset.get_pandas("train")
    causal_features = list(X_c.columns)

    X_ac, _, _, _ = arguablycausal_dataset.get_pandas("train")
    arguablycausal_features = list(X_ac.columns)

    X_all, y_all, _, domains_all = all_dataset.get_pandas("train")
    all_features = list(X_all.columns)
    anticausal_features = []
    if "anticausal" in experiment:
        anticausal_dataset = get_dataset(experiment['anticausal'], cache_dir)
        X_atc, _, _, _ = anticausal_dataset.get_pandas("train")
        anticausal_features = list(X_atc.columns)

    informative_features = list(set(arguablycausal_features + anticausal_features)) if anticausal_features else arguablycausal_features

    spurious_features = [feat for feat in all_features if feat not in informative_features]

    result = compute_conditional_confounding(X_all, y_all, domains_all,
                                               informative=informative_features,
                                               spurious=spurious_features)

    print(f"Experiment: {experiment}\nResult: {result}")
    with open('results_shift_measures.txt', "a") as f:
        f.write(f"Train Experiment: {experiment}\n{result}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache_dir", default="tmp", help="Directory to cache raw data files.")
    parser.add_argument("--debug", action="store_true", help="Run in debug mode.")
    parser.add_argument("--model", default="histgbm", help="Model to use.")
    args = parser.parse_args()

    experiments = [
        {"causal": "diabetes_readmission_causal", "arguablycausal": "diabetes_readmission_arguablycausal", "all": "diabetes_readmission"},
        {"causal": "acsfoodstamps_causal", "arguablycausal": "acsfoodstamps_arguablycausal", "all": "acsfoodstamps"},
        {"causal": "acsincome_causal", "arguablycausal": "acsincome_arguablycausal", "anticausal": "acsincome_anticausal", "all": "acsincome"},
        {"causal": "acspubcov_causal", "arguablycausal": "acspubcov_arguablycausal", "all": "acspubcov"},
        {"causal": "acsunemployment_causal", "arguablycausal": "acsunemployment_arguablycausal", "anticausal": "acsunemployment_anticausal", "all": "acsunemployment"},
        {"causal": "brfss_diabetes_causal", "arguablycausal": "brfss_diabetes_arguablycausal", "anticausal": "brfss_diabetes_anticausal", "all": "brfss_diabetes"},
        {"causal": "brfss_blood_pressure_causal", "arguablycausal": "brfss_blood_pressure_arguablycausal", "anticausal": "brfss_blood_pressure_anticausal", "all": "brfss_blood_pressure"},
        {"causal": "college_scorecard_causal", "arguablycausal": "college_scorecard_arguablycausal", "all": "college_scorecard"},
        {"causal": "assistments_causal", "arguablycausal": "assistments_arguablycausal", "all": "assistments"},
        {"causal": "nhanes_lead_causal", "arguablycausal": "nhanes_lead_arguablycausal", "all": "nhanes_lead"},
        {"causal": "physionet_causal", "arguablycausal": "physionet_arguablycausal", "all": "physionet"}
    ]

    for experiment in experiments:
        try:
            print(f"Running experiment: {experiment}")
            main(experiment, args.cache_dir, args.model, args.debug)
        except Exception as e:
            print(f"[ERROR] Experiment {experiment} failed: {e}")
            print(traceback.format_exc())