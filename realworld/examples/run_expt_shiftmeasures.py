import argparse
import logging

import torch
from sklearn.metrics import accuracy_score
import os
from tableshift import get_dataset
from tableshift.models.training import train
from tableshift.models.utils import get_estimator
from tableshift.models.torchutils import apply_model, get_module_attr,split_num_cat
from tableshift.models.default_hparams import get_default_config
from sklearn.model_selection import ParameterGrid

import random
import numpy as np
import torch
import pandas as pd
import npeet.entropy_estimators as ee
from sklearn.feature_selection import mutual_info_classif
from joblib import Parallel, delayed
import xgboost as xgb
from sklearn.decomposition import PCA
import os
from hps import cfgs_best_hps



LOG_LEVEL = logging.DEBUG

logger = logging.getLogger()
logging.basicConfig(
    format='%(asctime)s %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
    level=LOG_LEVEL,
    datefmt='%Y-%m-%d %H:%M:%S')

def compute_mi_terms(rep, hatY, Y_vals, E):
    try:
        ls = ee.mi(Y_vals, E)
        inf = ee.cmi(rep, Y_vals, E)    # informativeness
        inv = ee.cmi(rep, E, Y_vals)    # invariance
        lcs = ee.mi(rep, E)             # latent covariate shift (covariate shift)
        cs  = ee.cmi(Y_vals, E, rep)     # concept shift
        res = ee.cmi(rep, Y_vals, hatY)  # residual
        overall = inf - 0.5 * inv + 0.5 * ls + 0.5 * lcs - 0.5 * cs - res
        results = {
            "informativeness": inf,
            "invariance": inv,
            "label_shift": ls,
            "(latent)_covariate_shift": lcs,
            "concept_shift": cs,
            "residual": res,
            "overall": overall
        }
        return results
    except Exception as e:
        print("Error computing MI terms:", e)
        return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan
    
def subsample(X, y, e, n_samples=20000):
    idx = np.random.choice(len(y), min(n_samples, len(y)), replace=False)
    return X[idx], y[idx], e[idx]

def main(experiment, cache_dir, model, debug: bool):
    if debug:
        print("[INFO] running in debug mode.")
        experiment = "_debug"
    try:
        results_file = f"results/hyperparamsetting_besthps/shiftmeasures_{model}.txt"
        with open(results_file, "a") as f:
            log_dict = {f"{experiment}_{model}"}
            f.write(str(log_dict)+ "\n")

        dset = get_dataset(experiment, cache_dir)
        X, y, _, E = dset.get_pandas("train")
        config = get_default_config(model, dset)
        # if you want to use the default hyperparameters, you can skip this step
        best_hps = cfgs_best_hps[(model, experiment)]
        for key, value in best_hps.items():
                config[key] = value
        ################################################

        for ru in range(5):
            # Generate a random seed using Python's random module
            seed_value = np.random.randint(0, 10000)
            # Set the seed for NumPy
            np.random.seed(seed_value)

            # Set the seed for PyTorch
            torch.manual_seed(seed_value)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed_value)
                    
            estimator = get_estimator(model, **config)
            estimator = train(estimator, dset, config=config)
            # Get the predictions on the training set
            X_tensor = torch.tensor(X.values, dtype=torch.float32) if isinstance(X, pd.DataFrame) else X
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            X_tensor = X_tensor.to(device)
            # yhat = (estimator.predict(X_tensor) > 0.5).astype(int)

            # Compute mutual information terms
            # Y_vals = y.values if hasattr(y, 'values') else y

            if model in ['mlp', 'vrex', 'group_dro','irm']:
                # repeat the same process for the test set
                X_id_te, y_id_te, _, E_id_te = dset.get_pandas("id_test")
                X_id_te, y_id_te, E_id_te = subsample(X_id_te.values, y_id_te.values, E_id_te.values)
                X_id_te_tensor = torch.tensor(X_id_te, dtype=torch.float32)

                X_id_te_tensor = X_id_te_tensor.to(device)
                Y_vals_id_te = y_id_te.values if hasattr(y_id_te, 'values') else y_id_te
                yhat_id_te = (estimator.predict(X_id_te_tensor) > 0.5).astype(int)

                # Repeat the same process for the training set with model representations
                X_id_te_rep = estimator.get_activations(X_id_te_tensor).cpu().detach().numpy()
                mi_results_rep = compute_mi_terms(rep=X_id_te_rep, hatY=yhat_id_te, Y_vals=Y_vals_id_te, E=E_id_te)
                with open(results_file, "a") as f:
                    f.write("run_"+str(ru)+"Train Datashifts with Representations:" + str(mi_results_rep) + "\n")

                X_te, y_te, _, E_te = dset.get_pandas("ood_test")
                X_te, y_te, E_te = subsample(X_te.values, y_te.values, E_te.values)
                X_te_tensor = torch.tensor(X_te, dtype=torch.float32)
                X_te_tensor = X_te_tensor.to(device)
                Y_vals_te = y_te.values if hasattr(y_te, 'values') else y_te
                yhat_te = (estimator.predict(X_te_tensor) > 0.5).astype(int)
                
                X_te_rep = estimator.get_activations(X_te_tensor).cpu().detach().numpy()
                mi_results_te_rep = compute_mi_terms(rep=X_te_rep, hatY=yhat_te, Y_vals=Y_vals_te, E=E_te)
                with open(results_file, "a") as f:
                    f.write("run_"+str(ru)+"Test Datashifts with Representations:" + str(mi_results_te_rep) + "\n")

            elif model == 'xgb':
                X_id_te, y_id_te, _, E_id_te = dset.get_pandas("id_test")
                X_id_te, y_id_te, E_id_te = subsample(X_id_te.values, y_id_te.values, E_id_te.values)
                X_id_te_tensor = torch.tensor(X_id_te, dtype=torch.float32)
                X_id_te_tensor = X_id_te_tensor.to(device)
                Y_vals_id_te = y_id_te.values if hasattr(y_id_te, 'values') else y_id_te
                yhat_id_te = (estimator.predict(X_id_te_tensor.cpu().numpy()) > 0.5).astype(int)
                
                booster = estimator.get_booster()
                dmat_te = xgb.DMatrix(X_id_te)

                # leaf_idx    = booster.predict(dmat_te, pred_leaf=True, strict_shape=False,validate_features=False)
                raw_margin  = booster.predict(dmat_te, output_margin=True, strict_shape=False,validate_features=False)
                shap_vals   = booster.predict(dmat_te, pred_contribs=True, strict_shape=False,validate_features=False)

                X_id_te_rep = np.hstack([raw_margin.reshape(-1,1), shap_vals])
                
                mi_results_rep = compute_mi_terms(rep=X_id_te_rep, hatY=yhat_id_te, Y_vals=Y_vals_id_te, E=E_id_te)

                with open(results_file, "a") as f:
                    f.write("run_"+str(ru)+"_Train Datashifts with Representations:" + str(mi_results_rep) + "\n")

                X_te, y_te, _, E_te = dset.get_pandas("ood_test")
                X_te, y_te, E_te = subsample(X_te.values, y_te.values, E_te.values)
                X_te_tensor = torch.tensor(X_te, dtype=torch.float32)
                X_te_tensor = X_te_tensor.to(device)
                Y_vals_te = y_te.values if hasattr(y_te, 'values') else y_te
                yhat_te = (estimator.predict(X_te_tensor.cpu().numpy()) > 0.5).astype(int)
                dmat_te = xgb.DMatrix(X_te)
                # leaf_idx    = booster.predict(dmat_te, pred_leaf=True, strict_shape=False,validate_features=False)
                raw_margin  = booster.predict(dmat_te, output_margin=True, strict_shape=False,validate_features=False)
                shap_vals   = booster.predict(dmat_te, pred_contribs=True, strict_shape=False,validate_features=False)
                X_te_rep = np.hstack([raw_margin.reshape(-1,1), shap_vals])
                
                mi_results_te_rep = compute_mi_terms(rep=X_te_rep, hatY=yhat_te, Y_vals=Y_vals_te, E=E_te)
                with open(results_file, "a") as f:
                    f.write("run_"+str(ru)+"_Test Datashifts with Representations:" + str(mi_results_te_rep) + "\n")
                

            if not isinstance(estimator, torch.nn.Module):
                # Case: non-pytorch estimator; perform test-split evaluation.
                test_split = "ood_test" if dset.is_domain_split else "test"
                id_test_split = "id_test" if dset.is_domain_split else "test"
                # Fetch predictions and labels for a sklearn model.
                X_te, y_te, _, _ = dset.get_pandas(test_split)
                yhat_te = estimator.predict(X_te)

                X_id_te, y_id_te, _, _ = dset.get_pandas(id_test_split)
                yhat_id_te = estimator.predict(X_id_te)
            
                acc = accuracy_score(y_true=y_te, y_pred=yhat_te)
                acc_id = accuracy_score(y_true=y_id_te, y_pred=yhat_id_te)

                print(f"training completed! {test_split} accuracy: {acc:.4f}")
                print(f"training completed! {id_test_split} accuracy: {acc_id:.4f}")
                with open(results_file, "a") as f:
                    log_dict = {"ood_test": acc}
                    log_dict_id = {"id_test": acc_id}
                    f.write("run_"+str(seed_value)+"_ood_test accuracy: " + str(log_dict) + "\n")
                    f.write("run_"+str(seed_value)+"_id_test accuracy: " + str(log_dict_id) + "\n")
            else:
                # Case: pytorch estimator; eval is already performed + printed by train().
                print("training completed!")
        return
    except Exception as e:
        print(e)
        print(f"error in model {model}: {e}")
        return

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache_dir", default="tmp",
                        help="Directory to cache raw data files to.")
    parser.add_argument("--debug", action="store_true", default=False,
                        help="Whether to run in debug mode. If True, various "
                             "truncations/simplifications are performed to "
                             "speed up experiment.")
    parser.add_argument("--experiment", default="diabetes_readmission",
                        help="Experiment to run. Overridden when debug=True.")
    parser.add_argument("--model", default="histgbm",
                        help="model to use.")
    args = parser.parse_args()
    main(**vars(args))
