import os
import json
import numpy as np
import itertools
import matplotlib.pyplot as plt
import matplotlib as mpl

# --- Configuration ---------------------------------------------------------
models = ['xgb', 'mlp', 'gdro', 'irm', 'vrex']
titles = ['XGB', 'MLP', 'GDRO', 'IRM', 'VREX']
settings = ['causal', 'arguablycausal', 'all']
json_folder = 'results/hyperparamsetting'
dataset_titles = ['Readmission', 'Food Stamps', 'Diabetes', 'Income', 'Unemployment',
             'ASSISTments', 'Public Coverage', 'Hypertenson']
# --- Load data -------------------------------------------------------------
# raw_data[model] -> JSON structure
raw_data = {}
for model in models:
    path = os.path.join(json_folder, f'shiftmeasures_{model}.json')
    if not os.path.exists(path):
        print(f"Warning: missing {path}, skipping {model}")
        continue
    with open(path, 'r') as f:
        raw_data[model] = json.load(f)
# import pdb; pdb.set_trace()
# --- Compute metrics -------------------------------------------------------
# metrics[dataset][setting][model] = {train_inform_residual, test_inform_residual, train_formula, test_formula}
metrics = {}
dataset_metrics = {}
for model, data in raw_data.items():
    for dataset, ds in data.items():
        metrics.setdefault(dataset, {})
        dataset_metrics.setdefault(dataset, {})
        for setting in settings:
            if setting not in ds:
                continue
            id_test = np.array(ds[setting]['metrics']['id_test']).mean()
            id_test_std = np.array(ds[setting]['metrics']['id_test']).std()
            ood_test = np.array(ds[setting]['metrics']['ood_test']).mean()
            ood_test_std = np.array(ds[setting]['metrics']['ood_test']).std()
            ts = ds[setting]['train_shifts']
            tts = ds[setting]['test_shifts']
            train_ir = (np.array(ts['informativeness']) - np.array(ts['residual'])).mean()
            train_ir_std = (np.array(ts['informativeness']) - np.array(ts['residual'])).std()
            test_ir  = (np.array(tts['informativeness']) - np.array(tts['residual'])).mean()
            test_ir_std  = (np.array(tts['informativeness']) - np.array(tts['residual'])).std()
            train_f  = ((np.array(ts['label_shift'])/2 - np.array(ts['invariance'])/2
                        - np.array(ts['concept_shift'])/2 + np.array(ts['latent_covariate_shift'])/2)).mean()
            train_f_std  = ((np.array(ts['label_shift'])/2 - np.array(ts['invariance'])/2
                        - np.array(ts['concept_shift'])/2 + np.array(ts['latent_covariate_shift'])/2)).std()
            test_f   = (np.array(tts['label_shift'])/2 - np.array(tts['invariance'])/2
                        - np.array(tts['concept_shift'])/2 + np.array(tts['latent_covariate_shift'])/2).mean()
            test_f_std   = (np.array(tts['label_shift'])/2 - np.array(tts['invariance'])/2
                        - np.array(tts['concept_shift'])/2 + np.array(tts['latent_covariate_shift'])/2).std()
            metrics[dataset].setdefault(setting, {})[model] = {
                'id_test':              id_test,
                'ood_test':             ood_test,
                'train_inform_residual': train_ir,
                'test_inform_residual':  test_ir,
                'train_formula':         train_f,
                'test_formula':          test_f
            }
            dataset_metrics[dataset].setdefault(setting, {})[model] = {
                'id_test':              id_test,
                'id_test_std':          id_test_std,
                'ood_test':             ood_test,
                'ood_test_std':         ood_test_std,
                'train_inform_residual': train_ir,
                'train_inform_residual_std': train_ir_std,
                'test_inform_residual':  test_ir,
                'test_inform_residual_std':  test_ir_std,
                'train_formula':         train_f,
                'train_formula_std':    train_f_std,
                'test_formula':          test_f,
                'test_formula_std':     test_f_std
            }


# --- Average across datasets -----------------------------------------------
avg_metrics = {setting: {model: {
                        'id_test':              0,
                        'ood_test':             0,
                        'train_inform_residual': 0,
                        'test_inform_residual':  0,
                        'train_formula':         0,
                        'test_formula':          0}
                           for model in models}
               for setting in settings}

counts = {setting: 0 for setting in settings}
for setting in settings:
    for ds_vals in metrics.values():
        if setting in ds_vals:
            counts[setting] += 1
            for model in models:
                m = ds_vals[setting].get(model)
                if m:
                    for k, v in m.items():
                        avg_metrics[setting][model][k] += v
    # normalize
    if counts[setting] > 0:
        for model in models:
            for k in avg_metrics[setting][model]:
                avg_metrics[setting][model][k] /= counts[setting]

def sign_consistency(data):
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
                train_diff = np.array(data[t]["train_shifts"][measure]).mean() - np.array(data[s]["train_shifts"][measure]).mean()
                test_diff  = np.array(data[t]["test_shifts"][measure]).mean()  - np.array(data[s]["test_shifts"][measure]).mean()
                if train_diff * sign > 0:
                    count += 1
                if test_diff * sign > 0:
                    count += 1
            scores[measure] = count / total_checks
        return scores

    # 4) compute per‐dataset consistency
    for model in models:
        print(f"Model: {model}")
        per_dataset = {}
        path = os.path.join(json_folder, f'shiftmeasures_{model}.json')
        if not os.path.exists(path):
            print(f"Warning: missing {path}, skipping {model}")
            continue
        with open(path, 'r') as f:
            model_data = json.load(f)
            for name, dataset in model_data.items():
                per_dataset[name] = compute_consistency_all_pairs(dataset, sign_map)

        # 5) compute average consistency across datasets
        average_consistency = {}
        for measure in sign_map:
            vals = [per_dataset[d][measure] for d in per_dataset]
            average_consistency[measure] = sum(vals) / len(vals)
            print(f"  {measure:25s}: {average_consistency[measure]:.2f}")

def plot_all_train_test_combined():
    """
    Single figure with two subplots for the 'all' setting:
    left = train metrics, right = test metrics.
    Each subplot compares informativeness-residual vs formula across models.
    """
    setting = 'all'
    x = np.arange(len(models))
    width = 0.35
    # Values
    train_ir = [avg_metrics[setting][m]['train_inform_residual'] for m in models]
    train_f  = [avg_metrics[setting][m]['train_formula']         for m in models]
    test_ir  = [avg_metrics[setting][m]['test_inform_residual']  for m in models]
    test_f   = [avg_metrics[setting][m]['test_formula']          for m in models]
    # Create 1x2 subplots
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Train subplot
    ax = axes[0]
    ax.bar(x - width/2, train_ir, width, label=r'$\mathbf{CI - Res}$')
    ax.bar(x + width/2, train_f,  width, label=r'$\mathbf{-\frac{Var}{2} + \frac{LS}{2} + \frac{FS}{2} - \frac{CS}{2}}$')
    ax.set_xticks(x,fontweight='bold')
    ax.set_xticklabels(titles,fontweight='bold',fontsize=16)
    ax.set_title('In-domain: Avg. Over Datasets',fontweight='bold', fontsize=19)
    legend = ax.legend(fontsize=22, handleheight=0.2, handlelength=1.5, title_fontproperties={'weight': 'bold'})
    legend.get_texts()[0].set_color('#1f77b4')
    legend.get_texts()[1].set_color('#ff7f0e')
    plt.setp(ax.get_yticklabels(), fontweight='bold')

    # Test subplot
    ax = axes[1]
    ax.bar(x - width/2, test_ir, width, label=r'$\mathbf{CI - Res}$')
    ax.bar(x + width/2, test_f,  width, label=r'$\mathbf{-\frac{Var}{2} + \frac{LS}{2} + \frac{FS}{2} - \frac{CS}{2}}$')
    ax.set_xticks(x,fontweight='bold')
    ax.set_xticklabels(titles,fontweight='bold',fontsize=16)
    ax.set_title('Out-of-domain: Avg. Over Datasets',fontweight='bold', fontsize=19)
    legend = ax.legend(fontsize=22, handleheight=0.2, handlelength=1.5, title_fontproperties={'family':'sans-serif', 'weight': 'bold'})
    legend.get_texts()[0].set_color('#1f77b4')
    legend.get_texts()[1].set_color('#ff7f0e')
    plt.setp(ax.get_yticklabels(), fontweight='bold')

    # handles, labels = axes[0].get_legend_handles_labels()
    # fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, -0.075), ncol=4, fontsize=18, markerscale=2)

    plt.tight_layout()
    # plt.show()
    plt.savefig('results/hyperparamsetting/shiftdifferences.pdf', dpi=300, bbox_inches='tight')

def plot_all_datasets_all_setting():
    """
    Creates two figures (train & test) for the 'all' setting,
    each with 2x4 subplots—one subplot per dataset—comparing
    informativeness-residual and formula metrics across models.
    """
    # Filter datasets that have the 'all' setting
    datasets = [d for d in dataset_metrics if 'all' in dataset_metrics[d]]
    # print(datasets)

    # --- Train figure ---
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    for ax, dataset in zip(axes.flatten(), datasets):
        x = np.arange(len(models))
        width = 0.35
        mvals = dataset_metrics[dataset]['all']
        train_ir = [mvals[m]['train_inform_residual'] for m in models]
        train_ir_std = [mvals[m]['train_inform_residual_std'] for m in models]
        train_f  = [mvals[m]['train_formula']         for m in models]
        train_f_std  = [mvals[m]['train_formula_std']         for m in models]
        ax.bar(x - width/2, train_ir, width, yerr=train_ir_std, capsize=5, label=r'$\mathbf{CI - Res}$')
        ax.bar(x + width/2, train_f,  width, yerr=train_f_std, capsize=5, label=r'$\mathbf{-\frac{Var}{2} + \frac{LS}{2} + \frac{FS}{2} - \frac{CS}{2}}$')
        ax.set_title(dataset_titles[datasets.index(dataset)],fontweight='bold',fontsize=16)
        ax.set_xticks(x,fontweight='bold')
        ax.set_xticklabels(titles,fontweight='bold',fontsize=14)
        plt.setp(ax.get_yticklabels(), fontweight='bold')
    fig.suptitle("Train Environments",fontweight='bold',fontsize=18)
    handles, labels = axes[0,0].get_legend_handles_labels()
    legend=fig.legend(handles, labels, loc='lower center', ncol=4, fontsize=26, markerscale=2,handleheight=0.2, handlelength=2,)
    legend.get_texts()[0].set_color('#1f77b4')
    legend.get_texts()[1].set_color('#ff7f0e')
    plt.tight_layout(rect=[0,0.12,0.95,1])
    plt.show()
    plt.savefig('results/hyperparamsetting/train_shiftdifferences_datasets.pdf', dpi=300, bbox_inches='tight')


    # --- Test figure ---
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    for ax, dataset in zip(axes.flatten(), datasets):
        x = np.arange(len(models))
        width = 0.35
        mvals = dataset_metrics[dataset]['all']
        test_ir = [mvals[m]['test_inform_residual'] for m in models]
        test_ir_std = [mvals[m]['test_inform_residual_std'] for m in models]
        test_f  = [mvals[m]['test_formula']          for m in models]
        test_f_std  = [mvals[m]['test_formula_std']          for m in models]
        ax.bar(x - width/2, test_ir, width, yerr=test_ir_std, capsize=5,label=r'$\mathbf{CI - Res}$')
        ax.bar(x + width/2, test_f,  width, yerr=test_f_std, capsize=5,label=r'$\mathbf{-\frac{Var}{2} + \frac{LS}{2} + \frac{FS}{2} - \frac{CS}{2}}$')
        ax.set_yticks(ax.get_yticks(), fontweight='bold')
        ax.set_title(dataset_titles[datasets.index(dataset)],fontweight='bold',fontsize=16)
        ax.set_xticks(x,fontweight='bold')
        ax.set_xticklabels(titles,fontweight='bold',fontsize=14)
        plt.setp(ax.get_yticklabels(), fontweight='bold')
    fig.suptitle("Test Environments",fontweight='bold',fontsize=18)
    handles, labels = axes[0,0].get_legend_handles_labels()
    legend = fig.legend(handles, labels, loc='lower center', ncol=4, fontsize=26, markerscale=2,handleheight=0.2, handlelength=2,)
    legend.get_texts()[0].set_color('#1f77b4')
    legend.get_texts()[1].set_color('#ff7f0e')
    plt.tight_layout(rect=[0,0.12,0.95,1])
    plt.show()
    plt.savefig('results/hyperparamsetting/test_shiftdifferences_datasets.pdf', dpi=300, bbox_inches='tight')
# # --- Invocation ------------------------------------------------------------


# plot_all_train_test_combined()
# plot_all_datasets_all_setting()
# for dataset, ds in dataset_metrics.items():
#     print(f"Dataset: {dataset}")
#     for setting, mvals in ds.items():
#         print(f"Setting: {setting}")
#         for model, m in mvals.items():
#             print(f"Model: {model}")
#             print(m['id_test'], m['id_test_std'])
#             print(m['ood_test'], m['ood_test_std'])


# for model in models:
#     print(f"Model: {model}")
#     for setting in settings:
#         print(setting, "id_test", avg_metrics[setting][model]['id_test'])
#         print(setting, "id_test", avg_metrics[setting][model]['ood_test'])

# sign_consistency(metrics)