# Code to load this JSON and create 8×7 train/test plots with error bars
import json
import matplotlib.pyplot as plt
import numpy as np

models = ['mlp', 'xgb', 'vrex', 'irm', 'gdro']
titles = ['MLP', 'XGB', 'VREX', 'IRM', 'GDRO']

def plot_results(model):
    # Load JSON
    with open('results/hyperparamsetting/shiftmeasures_'+model+'.json') as f:
        data = json.load(f)

    # Prepare data structures
    settings = ['causal', 'arguablycausal', 'all']
    shift_terms = ['informativeness', 'invariance', 'latent_covariate_shift', 
                   'label_shift', 'concept_shift', 'residual']
    shift_terms_titles = ['Cond. Inform.', 'Variation', 'Feature Shift', 
                         'Label Shift', 'Concept Shift', 'Residual', 'OOD Test ACC.']
    
    dataset_map = {
        "diabetes_readmission": "Readmission",
        "acsfoodstamps": "Food Stamps",
        "brfss_diabetes": "Diabetes",
        "acsincome": "Income",
        "assistments": "ASSISTments",
        "acspubcov": "Public Coverage",
        "brfss_blood_pressure": "Blood Pressure"
    }
    datasets = list(dataset_map.keys())
    metrics = shift_terms + ['ood_test']

    # Create figure grids
    fig_train, axes_train = plt.subplots(len(datasets), len(metrics), 
                                       figsize=(20, 3*len(datasets)), sharey='row')
    fig_test, axes_test = plt.subplots(len(datasets), len(metrics), 
                                     figsize=(20, 3*len(datasets)), sharey='row')

    # Plotting loop
    for i, ds in enumerate(datasets):
        for j, metric in enumerate(metrics):
            train_means, train_stds = [], []
            test_means, test_stds = [], []
            
            for s in settings:
                entry = data[ds].get(s, {})
                
                # Get values with array defaults
                if metric == 'ood_test':
                    train_vals = entry.get('metrics', {}).get('ood_test', [0])
                    test_vals = entry.get('metrics', {}).get('ood_test', [0])
                else:
                    train_vals = entry.get('train_shifts', {}).get(metric, [0])
                    test_vals = entry.get('test_shifts', {}).get(metric, [0])
                
                # Calculate statistics
                train_means.append(np.mean(train_vals))
                train_stds.append(np.std(train_vals))
                test_means.append(np.mean(test_vals))
                test_stds.append(np.std(test_vals))

            xs = range(len(settings))
            
            # Plot training data
            ax = axes_train[i, j]
            ax.bar(xs, train_means, yerr=train_stds, capsize=5, alpha=0.7)
            ax.set_xticks(xs)
            ax.set_xticklabels(['C', 'A.C', 'A'], fontsize=16, fontweight='bold')
            if j == 0:
                ax.set_ylabel(dataset_map[ds], fontsize=16, fontweight='bold')
            if i == 0:
                ax.set_title(shift_terms_titles[j], fontsize=16, pad=10, fontweight='bold')
            plt.setp(ax.get_yticklabels(), fontweight='bold')
            plt.setp(ax.get_xticklabels(), fontweight='bold')

            # Plot test data
            ax2 = axes_test[i, j]
            ax2.bar(xs, test_means, yerr=test_stds, capsize=5, alpha=0.7)
            ax2.set_xticks(xs)
            ax2.set_xticklabels(['C', 'A.C', 'A'], fontsize=16, fontweight='bold')
            if j == 0:
                ax2.set_ylabel(dataset_map[ds], fontsize=16, fontweight='bold')
            if i == 0:
                ax2.set_title(shift_terms_titles[j], fontsize=16, pad=10, fontweight='bold')
            plt.setp(ax2.get_yticklabels(), fontweight='bold')
            plt.setp(ax2.get_xticklabels(), fontweight='bold')
    # Final formatting
    fig_train.suptitle(f'Train Data Shifts - {titles[models.index(model)]}', 
                      fontsize=20, y=0.98, fontweight='bold')
    fig_test.suptitle(f'Test Data Shifts - {titles[models.index(model)]}', 
                     fontsize=20, y=0.98, fontweight='bold')
    
    fig_train.tight_layout(rect=[0, 0, 1, 0.97])
    fig_test.tight_layout(rect=[0, 0, 1, 0.97])
    
    fig_train.savefig(f'results/hyperparamsetting/train_datashifts_{model}.pdf', 
                     bbox_inches='tight', dpi=300)
    fig_test.savefig(f'results/hyperparamsetting/test_datashifts_{model}.pdf', 
                    bbox_inches='tight', dpi=300)
    plt.close()

for name in models:
    plot_results(name)