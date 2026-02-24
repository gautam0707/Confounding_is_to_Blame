import os
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

# --- Configuration ---------------------------------------------------------
models = ['xgb', 'mlp', 'group_dro', 'irm', 'vrex']
titles = ['XGB', 'MLP', 'GDRO', 'IRM', 'VREX']
settings = ['causal', 'arguablycausal', 'all']
json_folder = 'results/informative'
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

# --- Compute metrics -------------------------------------------------------
# metrics[dataset][setting][model] = {train_inform_residual, test_inform_residual, train_formula, test_formula}
metrics = {}
for model, data in raw_data.items():
    for dataset, ds in data.items():
        metrics.setdefault(dataset, {})
        for setting in settings:
            if setting not in ds:
                continue
            ts = ds[setting]['train_shifts']
            tts = ds[setting]['test_shifts']
            train_ir = ts['informativeness'] - ts['residual']
            test_ir  = tts['informativeness'] - tts['residual']
            train_f  = (ts['label_shift']/2 - ts['invariance']/2
                        - ts['concept_shift']/2 + ts['latent_covariate_shift']/2)
            test_f   = (tts['label_shift']/2 - tts['invariance']/2
                        - tts['concept_shift']/2 + tts['latent_covariate_shift']/2)
            # import pdb; pdb.set_trace()
            metrics[dataset].setdefault(setting, {})[model] = {
                'train_inform_residual': train_ir,
                'test_inform_residual':  test_ir,
                'train_formula':         train_f,
                'test_formula':          test_f
            }

# --- Average across datasets -----------------------------------------------
avg_metrics = {setting: {model: {'train_inform_residual': 0,
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

# --- Plotting functions ----------------------------------------------------
# def plot_overall(which='train'):
#     """
#     Plot average metrics (inform_residual vs formula) for all models
#     across the three settings, for either 'train' or 'test'.
#     """
#     fig, axes = plt.subplots(1, 3, figsize=(15, 4))
#     for i, setting in enumerate(settings):
#         ax = axes[i]
#         x = np.arange(len(models))
#         width = 0.35
#         ir = [avg_metrics[setting][m][f'{which}_inform_residual'] for m in models]
#         fr = [avg_metrics[setting][m][f'{which}_formula'] for m in models]
#         ax.bar(x - width/2, ir, width, label=r'$Conditional\ Informativeness - Residual$')
#         ax.bar(x + width/2, fr, width, label=r'$\frac{Label\ Shift}{2} - \frac{Invariance}{2} - \frac{Concept\ Shift}{2} + \frac{Feature\ Shift}{2}$')
#         ax.set_xticks(x)
#         ax.set_xticklabels(titles)
#         ax.set_title(f"{setting} ({which})")
#         ax.legend()
#     fig.suptitle(f'Average {which.capitalize()} Metrics by Setting and Model')
#     plt.tight_layout()
#     # plt.show()
#     plt.savefig(f'results/informative/average_{which}_metrics.pdf', dpi=300, bbox_inches='tight')
#     plt.close()


def plot_per_dataset(dataset, which='train'):
    """
    Plot metrics for a single dataset across the three settings,
    for either 'train' or 'test'.
    """
    ds_vals = metrics[dataset]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for i, setting in enumerate(settings):
        ax = axes[i]
        x = np.arange(len(models))
        width = 0.35
        ir = [ds_vals.get(setting, {}).get(m, {}).get(f'{which}_inform_residual', np.nan)
              for m in models]
        fr = [ds_vals.get(setting, {}).get(m, {}).get(f'{which}_formula', np.nan)
              for m in models]
        ax.bar(x - width/2, ir, width)
        ax.bar(x + width/2, fr, width)
        ax.set_xticks(x)
        ax.set_xticklabels(titles)
        ax.set_title(f"{setting} ({which})")
    fig.suptitle(f'{dataset} — {which.capitalize()} Metrics')
    plt.tight_layout()
    # plt.show()
    plt.savefig(f'results/informative/{dataset}_{which}_metrics.pdf', dpi=300, bbox_inches='tight')
    plt.close()

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
    plt.savefig('results/informative/shiftdifferences.pdf', dpi=300, bbox_inches='tight')

def plot_all_datasets_all_setting():
    """
    Creates two figures (train & test) for the 'all' setting,
    each with 2x4 subplots—one subplot per dataset—comparing
    informativeness-residual and formula metrics across models.
    """
    # Filter datasets that have the 'all' setting
    datasets = [d for d in metrics if 'all' in metrics[d]]
    # print(datasets)

    # --- Train figure ---
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    for ax, dataset in zip(axes.flatten(), datasets):
        x = np.arange(len(models))
        width = 0.35
        mvals = metrics[dataset]['all']
        train_ir = [mvals[m]['train_inform_residual'] for m in models]
        train_f  = [mvals[m]['train_formula']         for m in models]
        ax.bar(x - width/2, train_ir, width, label=r'$\mathbf{CI - Res}$')
        ax.bar(x + width/2, train_f,  width, label=r'$\mathbf{-\frac{Var}{2} + \frac{LS}{2} + \frac{FS}{2} - \frac{CS}{2}}$')
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
    plt.savefig('results/informative/train_shiftdifferences_datasets.pdf', dpi=300, bbox_inches='tight')


    # --- Test figure ---
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    for ax, dataset in zip(axes.flatten(), datasets):
        x = np.arange(len(models))
        width = 0.35
        mvals = metrics[dataset]['all']
        test_ir = [mvals[m]['test_inform_residual'] for m in models]
        test_f  = [mvals[m]['test_formula']          for m in models]
        ax.bar(x - width/2, test_ir, width,label=r'$\mathbf{CI - Res}$')
        ax.bar(x + width/2, test_f,  width,label=r'$\mathbf{-\frac{Var}{2} + \frac{LS}{2} + \frac{FS}{2} - \frac{CS}{2}}$')
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
    plt.savefig('results/informative/test_shiftdifferences_datasets.pdf', dpi=300, bbox_inches='tight')
# --- Invocation ------------------------------------------------------------
# To produce the two overview figures:
# plot_overall('train')
# plot_overall('test')

# To produce dataset-specific figures for all datasets:
# for ds in metrics:
#     plot_per_dataset(ds, 'train')
#     plot_per_dataset(ds, 'test')

plot_all_train_test_combined()
plot_all_datasets_all_setting()