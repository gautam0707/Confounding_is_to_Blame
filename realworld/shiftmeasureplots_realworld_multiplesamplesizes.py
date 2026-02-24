import numpy as np
import matplotlib.pyplot as plt

import os
import json

path = os.path.join('results/informative', f'shiftmeasures_xgb_multiplesamplesizes.json')
data = None
with open(path, 'r') as f:
    data = json.load(f)


# Compute metrics for plotting
sample_sizes = data["Train"].keys()
train_metric1 = [data["Train"][n]["informativeness"] - data["Train"][n]["residual"] for n in sample_sizes]
train_metric2 = [
    data["Train"][n]["label_shift"] + data["Train"][n]["latent_covariate_shift"]
    - data["Train"][n]["invariance"] - data["Train"][n]["concept_shift"]
    for n in sample_sizes
]
test_metric1 = [data["Test"][n]["informativeness"] - data["Test"][n]["residual"] for n in sample_sizes]
test_metric2 = [
    data["Test"][n]["label_shift"] + data["Test"][n]["latent_covariate_shift"]
    - data["Test"][n]["invariance"] - data["Test"][n]["concept_shift"]
    for n in sample_sizes
]

# Setup for side-by-side bars
x = np.arange(len(sample_sizes))
width = 0.35

fig, axes = plt.subplots(1, 2, figsize=(10, 5), sharey=True)

# Train subplot
axes[0].bar(x - width/2, train_metric1, width, label=r'$\mathbf{CI - Res}$')
axes[0].bar(x, train_metric2, width, label=r'$\mathbf{-\frac{Var}{2} + \frac{LS}{2} + \frac{FS}{2} - \frac{CS}{2}}$')
axes[0].set_xticks(x)
axes[0].set_xticklabels([str(n) for n in sample_sizes])
axes[0].set_title('ID, acsincome dataset, XGB model', fontweight='bold', fontsize=14)
axes[0].set_xlabel('Sample Size', fontweight='bold', fontsize=14)
plt.setp(axes[0].get_yticklabels(), fontweight='bold')
plt.setp(axes[0].get_xticklabels(), fontweight='bold')


# Test subplot
axes[1].bar(x - width/2, test_metric1, width, label=r'$\mathbf{CI - Res}$')
axes[1].bar(x, test_metric2, width, label=r'$\mathbf{-\frac{Var}{2} + \frac{LS}{2} + \frac{FS}{2} - \frac{CS}{2}}$')
axes[1].set_xticks(x)
axes[1].set_xticklabels([str(n) for n in sample_sizes])
axes[1].set_title('OOD, acsincome dataset, XGB model', fontweight='bold', fontsize=14)
axes[1].set_xlabel('Sample Size', fontweight='bold', fontsize=14)
plt.setp(axes[1].get_yticklabels(), fontweight='bold')
plt.setp(axes[1].get_xticklabels(), fontweight='bold')

handles, labels = axes[0].get_legend_handles_labels()
legend=fig.legend(handles, labels, loc='lower center', ncol=4, fontsize=22, markerscale=2,handleheight=0.2, handlelength=2,)
legend.get_texts()[0].set_color('#1f77b4')
legend.get_texts()[1].set_color('#ff7f0e')

plt.tight_layout(rect=[0,0.15,0.95,1])
plt.show()
# Save the figure
fig.savefig('results/informative/shiftmeasures_xgb_multiplesamplesizes.pdf', dpi=300, bbox_inches='tight')
plt.close()

