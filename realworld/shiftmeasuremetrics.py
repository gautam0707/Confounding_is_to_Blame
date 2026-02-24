import numpy as np
import itertools
import json

from collections import defaultdict

models = ['xgb','mlp','group_dro','irm','vrex']
for model in models:
    print(f"Processing model: {model}")
    with open('results/informative/shiftmeasures_'+model+'.json') as f:
        data = json.load(f)

    # --- 2) define expected sign for each measure (+1 means should increase with accuracy; -1 means should decrease) ---
    sign_map = {
        "informativeness": +1,
        "latent_covariate_shift": +1,
        "invariance": -1,
        "concept_shift": -1,
        "residual": -1,
    }

    # 3) function that checks all pairs of settings
    def compute_consistency_all_pairs(data, sign_map):
        settings = ["causal", "arguablycausal", "all"]
        pairs = list(itertools.combinations(settings, 2))
        total_checks = 2 * len(pairs)  # train + test for each pair

        scores = {}
        for measure, sign in sign_map.items():
            count = 0
            for (s, t) in pairs:
                train_diff = data[t]["train_shifts"][measure] - data[s]["train_shifts"][measure]
                test_diff  = data[t]["test_shifts"][measure]  - data[s]["test_shifts"][measure]
                if train_diff * sign > 0:
                    count += 1
                if test_diff * sign > 0:
                    count += 1
            scores[measure] = count / total_checks
        return scores

    # 4) compute per‐dataset consistency
    per_dataset = {}
    for name, dataset in data.items():
        per_dataset[name] = compute_consistency_all_pairs(dataset, sign_map)

    # 5) compute average consistency across datasets
    average_consistency = {}
    for measure in sign_map:
        vals = [per_dataset[d][measure] for d in per_dataset]
        average_consistency[measure] = sum(vals) / len(vals)

    # # 6) display results
    # print("Per-dataset consistency:")
    # for name, scores in per_dataset.items():
    #     print(f"\nDataset: {name}")
    #     for m, v in scores.items():
    #         print(f"  {m:25s}: {v:.2f}")



    ood_test_accuracies = {
        "causal": [],
        "arguablycausal": [],
        "all": []
    }
    id_test_accuracies = {
        "causal": [],
        "arguablycausal": [],
        "all": []
    }

    # Aggregate ood_test accuracies for each setting
    for dataset_name, dataset_results in data.items():
        for setting in ["causal", "arguablycausal", "all"]:
            if setting in dataset_results:
                try:
                    acc = dataset_results[setting]["metrics"]["ood_test"]
                    ood_test_accuracies[setting].append(acc)

                    id_acc = dataset_results[setting]["metrics"]["id_test"]
                    id_test_accuracies[setting].append(id_acc)
                except KeyError:
                    print(f"Missing 'ood_test' in {dataset_name} under {setting}")
    # Compute and print averages
    # for setting in ["causal", "arguablycausal", "all"]:
    #     accs = ood_test_accuracies[setting]
    #     id_accs = id_test_accuracies[setting]
    #     avg = sum(accs) / len(accs) if accs else 0
    #     id_avg = sum(id_accs) / len(id_accs) if id_accs else 0
    #     print(f"Average ID Test Accuracy ({setting}): {id_avg:.4f}")
    #     print(f"Average OOD Test Accuracy ({setting}): {avg:.4f}")

    # Compute and store additional results
    additional_results = {}  # {dataset_name: {setting: {metrics...}}}
    for dataset_name, dataset_results in data.items():
        additional_results[dataset_name] = {}
        for setting in ["causal", "arguablycausal", "all"]:
            if setting in dataset_results:
                try:
                    train_shifts = dataset_results[setting]["train_shifts"]
                    test_shifts = dataset_results[setting]["test_shifts"]

                    train_inform_residual = train_shifts["informativeness"] - train_shifts["residual"]
                    test_inform_residual = test_shifts["informativeness"] - test_shifts["residual"]

                    train_formula = (
                        train_shifts["label_shift"] / 2
                        - train_shifts["invariance"] / 2
                        - train_shifts["concept_shift"] / 2
                        + train_shifts["latent_covariate_shift"] / 2
                    )
                    test_formula = (
                        test_shifts["label_shift"] / 2
                        - test_shifts["invariance"] / 2
                        - test_shifts["concept_shift"] / 2
                        + test_shifts["latent_covariate_shift"] / 2
                    )

                    additional_results[dataset_name][setting] = {
                        "train_inform_residual": train_inform_residual,
                        "test_inform_residual": test_inform_residual,
                        "train_formula": train_formula,
                        "test_formula": test_formula,
                    }
                except KeyError as e:
                    print(f"Missing key {e} in {dataset_name} under {setting}")

    # Print per-dataset results
    # print("\nPer-Dataset Results:")
    # for dataset_name, dataset_info in additional_results.items():
    #     print(f"\nDataset: {dataset_name}")
    #     for setting, metrics in dataset_info.items():
    #         print(f"  Setting: {setting}")
    #         print(f"    Train (informativeness - residual): {metrics['train_inform_residual']:.4f}")
    #         print(f"    Test  (informativeness - residual): {metrics['test_inform_residual']:.4f}")
    #         print(f"    Train (formula result): {metrics['train_formula']:.4f}")
    #         print(f"    Test  (formula result): {metrics['test_formula']:.4f}")

    # Now compute and print averages PER SETTING across datasets
    # print("\nAveraged Results Across All Datasets (per setting):")
    for setting in ["causal", "arguablycausal", "all"]:
        setting_totals = {
            "train_inform_residual": 0,
            "test_inform_residual": 0,
            "train_formula": 0,
            "test_formula": 0,
        }
        setting_count = 0
        for dataset_name, results in additional_results.items():
            if setting in results:
                setting_count += 1
                setting_totals["train_inform_residual"] += results[setting]["train_inform_residual"]
                setting_totals["test_inform_residual"] += results[setting]["test_inform_residual"]
                setting_totals["train_formula"] += results[setting]["train_formula"]
                setting_totals["test_formula"] += results[setting]["test_formula"]

        # if setting_count > 0:
        #     print(f"\n  Setting: {setting}")
        #     print(f"    Train (informativeness - residual): {setting_totals['train_inform_residual'] / setting_count:.4f}")
        #     print(f"    Test  (informativeness - residual): {setting_totals['test_inform_residual'] / setting_count:.4f}")
        #     print(f"    Train (formula result): {setting_totals['train_formula'] / setting_count:.4f}")
        #     print(f"    Test  (formula result): {setting_totals['test_formula'] / setting_count:.4f}")

    for setting in ["causal", "arguablycausal", "all"]:
        accs = ood_test_accuracies[setting]
        id_accs = id_test_accuracies[setting]
        id_avg = sum(id_accs) / len(id_accs) if id_accs else 0
        print(f"Average ID Test Accuracy ({setting}): {id_avg:.4f}")
        avg = sum(accs) / len(accs) if accs else 0
        print(f"Average OOD Test Accuracy ({setting}): {avg:.4f}")

    # print("\nAverage consistency across all datasets:")
    # for m, v in average_consistency.items():
    #     print(f"  {m:25s}: {v:.2f}")