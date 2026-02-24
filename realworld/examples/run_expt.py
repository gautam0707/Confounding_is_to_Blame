import argparse
import logging

import torch
from sklearn.metrics import accuracy_score
import os
from tableshift import get_dataset
from tableshift.models.training import train
from tableshift.models.utils import get_estimator
from tableshift.models.default_hparams import get_default_config
import random
import numpy as np
import torch

# Generate a random seed using Python's random module
seed_value = random.randint(0, 2**32 - 1)

# Set the seed for NumPy
np.random.seed(seed_value)

# Set the seed for PyTorch
torch.manual_seed(seed_value)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed_value)


LOG_LEVEL = logging.DEBUG

logger = logging.getLogger()
logging.basicConfig(
    format='%(asctime)s %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
    level=LOG_LEVEL,
    datefmt='%Y-%m-%d %H:%M:%S')


def main(experiment, cache_dir, model, debug: bool):
    if debug:
        print("[INFO] running in debug mode.")
        experiment = "_debug"
    try:
        results_file = "results/informative/setting2_seed3_nonlinear.txt"
        with open(results_file, "a") as f:
            f.write("Model: "+model + "\n")
        dset = get_dataset(experiment, cache_dir)
        X, y, _, _ = dset.get_pandas("train")
        config = get_default_config(model, dset)
        if model in ['histgbm']:
            del config['batch_size']
            del config['n_epochs']
        estimator = get_estimator(model, **config)
        estimator = train(estimator, dset, config=config)
        
        if not isinstance(estimator, torch.nn.Module):
            # Case: non-pytorch estimator; perform test-split evaluation.
            test_split = "ood_test" if dset.is_domain_split else "test"
            # Fetch predictions and labels for a sklearn model.
            X_te, y_te, _, _ = dset.get_pandas(test_split)
            yhat_te = estimator.predict(X_te)
        
            acc = accuracy_score(y_true=y_te, y_pred=yhat_te)
            print(f"training completed! {test_split} accuracy: {acc:.4f}")
            with open(results_file, "a") as f:
                log_dict = {"ood_test": acc}
                f.write(str(log_dict)+ "\n")
        
        else:
            # Case: pytorch estimator; eval is already performed + printed by train().
            print("training completed!")
        return
    except Exception as e:
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
