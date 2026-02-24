import os
import json
import numpy as np
import itertools
import matplotlib.pyplot as plt
import matplotlib as mpl

# --- Configuration ---------------------------------------------------------
models = ['mlp', 'group_dro', 'irm']
titles = ['MLP', 'GDRO', 'IRM']
settings = ['all']
json_folder = 'results/hyperparamsetting_besthps'
dataset_titles = ['Readmission', 'Food Stamps', 'Diabetes', 'Income', 'Unemployment']
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
            train_overall = np.array(ts['overall']).mean()
            test_overall = np.array(tts['overall']).mean()
            train_overall_std = np.array(ts['overall']).std()
            test_overall_std = np.array(tts['overall']).std()

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
                'test_formula':          test_f,
                'train_overall':        train_overall,
                'test_overall':         test_overall,
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
                'test_formula_std':     test_f_std,
                'train_overall':        train_overall,
                'test_overall':         test_overall,
                'train_overall_std': train_overall_std,
                'test_overall_std': test_overall_std,
            }


# --- Average across datasets -----------------------------------------------
avg_metrics = {setting: {model: {
                        'id_test':              0,
                        'ood_test':             0,
                        'train_inform_residual': 0,
                        'test_inform_residual':  0,
                        'train_formula':         0,
                        'test_formula':          0,
                        'train_overall':        0,
                        'test_overall':         0,
                        }
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
    train_overall = [avg_metrics[setting][m]['train_overall'] for m in models]
    test_overall = [avg_metrics[setting][m]['test_overall'] for m in models]
    # Create 1x2 subplots
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Train subplot
    ax = axes[0]
    ax.bar(x - width, train_overall, width, label=r'$\mathbf{Predictive\ information}$', color='grey')
    ax.bar(x - width/2, train_ir, width, label=r'$\mathbf{Conditional\ informativeness - Residual}$')
    ax.bar(x, train_f,  width, label=r'$\mathbf{-\frac{Variation}{2} + \frac{Label\ shift}{2} + \frac{Feature\ shift}{2} - \frac{Concept\ shift}{2}}$')
    ax.set_xticks(x,fontweight='bold')
    ax.set_xticklabels(titles,fontweight='bold',fontsize=16)
    ax.set_title('In-domain: Avg. Over Datasets',fontweight='bold', fontsize=19)
    # legend = ax.legend(fontsize=16, handleheight=0.2, handlelength=1.5, title_fontproperties={'weight': 'bold'})
    # legend.get_texts()[0].set_color('grey')
    # legend.get_texts()[1].set_color('#1f77b4')
    # legend.get_texts()[2].set_color('#ff7f0e')
    plt.setp(ax.get_yticklabels(), fontweight='bold')

    # Test subplot
    ax = axes[1]
    ax.bar(x - width, test_overall, width, label=r'$\mathbf{Pred. Inform.}$',color='grey')
    ax.bar(x - width/2, test_ir, width, label=r'$\mathbf{CI - Res}$')
    ax.bar(x, test_f,  width, label=r'$\mathbf{-\frac{Var}{2} + \frac{LS}{2} + \frac{FS}{2} - \frac{CS}{2}}$')
    ax.set_xticks(x,fontweight='bold')
    ax.set_xticklabels(titles,fontweight='bold',fontsize=16)
    ax.set_title('Out-of-domain: Avg. Over Datasets',fontweight='bold', fontsize=19)
    # legend = ax.legend(fontsize=16, handleheight=0.2, handlelength=1.5, title_fontproperties={'family':'sans-serif', 'weight': 'bold'})
    # legend.get_texts()[0].set_color('grey')
    # legend.get_texts()[1].set_color('#1f77b4')
    # legend.get_texts()[2].set_color('#ff7f0e')
    plt.setp(ax.get_yticklabels(), fontweight='bold')

    handles, labels = axes[0].get_legend_handles_labels()
    legend = fig.legend(handles, labels, loc='lower center', ncol=2, fontsize=16, markerscale=2, handleheight=0.2, handlelength=2)
    legend.get_texts()[0].set_color('grey')
    legend.get_texts()[1].set_color('#1f77b4')
    legend.get_texts()[2].set_color('#ff7f0e')

    plt.tight_layout(rect=[0, 0.2, 1, 1])
    # plt.show()
    plt.savefig('results/hyperparamsetting_besthps/shiftdifferences.pdf', dpi=300, bbox_inches='tight')

def plot_all_datasets_all_setting():
    """
    Creates a single 2x6 figure for the 'all' setting,
    with train and test rows, each row comparing
    informativeness-residual and formula metrics across models.
    """
    datasets = [d for d in dataset_metrics if 'all' in dataset_metrics[d]]

    # Create a 2x6 figure
    fig, axes = plt.subplots(2, 5, figsize=(26, 10))

    # Plot train and test data in respective rows
    for row_idx, metric_type in enumerate(['train', 'test']):
        inform_residual_key = f'{metric_type}_inform_residual'
        inform_residual_std_key = f'{metric_type}_inform_residual_std'
        formula_key = f'{metric_type}_formula'
        formula_std_key = f'{metric_type}_formula_std'
        overall_key = f'{metric_type}_overall'
        overall_std_key = f'{metric_type}_overall_std'

        for col_idx, dataset in enumerate(datasets):
            ax = axes[row_idx, col_idx]
            x = np.arange(len(models))
            width = 0.35
            mvals = dataset_metrics[dataset]['all']
            ir = [mvals[m][inform_residual_key] for m in models]
            ir_std = [mvals[m][inform_residual_std_key] for m in models]
            f = [mvals[m][formula_key] for m in models]
            f_std = [mvals[m][formula_std_key] for m in models]
            overall = [mvals[m][overall_key] for m in models]
            overall_std = [mvals[m][overall_std_key] for m in models]
            
            ax.bar(x - width, overall, width, yerr=overall_std, capsize=5, label=r'$\mathbf{Predictive\ information}$',color='grey')
            ax.bar(x - width/2, ir, width, yerr=ir_std, capsize=5, label=r'$\mathbf{Conditional \ informativeness - Residual}$')
            ax.bar(x, f,  width, yerr=f_std, capsize=5, label=r'$\mathbf{-\frac{Variation}{2} + \frac{Label\ shift}{2} + \frac{Feature\ shift}{2} - \frac{Concept\ shift}{2}}$')
            ax.set_title(dataset_titles[datasets.index(dataset)], fontweight='bold', fontsize=16)
            ax.set_xticks(x)
            ax.set_xticklabels(titles, fontweight='bold', fontsize=14)
            plt.setp(ax.get_yticklabels(), fontweight='bold')
            if metric_type == 'test':
                ax.set_yticks(ax.get_yticks(), fontweight='bold')

    # Add row labels
    fig.text(-0.01, 0.75, 'Train Environments', rotation='vertical', fontweight='bold', fontsize=18, va='center')
    fig.text(-0.01, 0.35, 'Test Environments', rotation='vertical', fontweight='bold', fontsize=18, va='center')

    # Create a single legend at the bottom
    handles, labels = axes[0, 0].get_legend_handles_labels()
    legend = fig.legend(handles, labels, loc='lower center', ncol=4, fontsize=24, markerscale=2, handleheight=0.2, handlelength=2)
    legend.get_texts()[0].set_color('grey')
    legend.get_texts()[1].set_color('#1f77b4')
    legend.get_texts()[2].set_color('#ff7f0e')

    plt.tight_layout(rect=[0, 0.12, 1, 1])
    plt.show()
    plt.savefig('results/hyperparamsetting_besthps/combined_shiftdifferences_datasets.pdf', dpi=300, bbox_inches='tight')



plot_all_train_test_combined()
plot_all_datasets_all_setting()
for dataset, ds in dataset_metrics.items():
    print(f"Dataset: {dataset}")
    for setting, mvals in ds.items():
        print(f"Setting: {setting}")
        for model, m in mvals.items():
            print(f"Model: {model}")
            print(m['id_test'], m['id_test_std'])
            print(m['ood_test'], m['ood_test_std'])


for model in models:
    print(f"Model: {model}")
    for setting in settings:
        print(setting, "id_test", avg_metrics[setting][model]['id_test'])
        print(setting, "id_test", avg_metrics[setting][model]['ood_test'])
