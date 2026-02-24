# import matplotlib.pyplot as plt
# import numpy as np
# import ast

# # Function to read data from a text file
# def read_data_from_file(file_path):
#     with open(file_path, 'r') as file:
#         data = file.readlines()
#     return data

# # Function to process data and return results
# def process_data(data):
#     results = {}
#     current_model = None
    
#     for line in data:
#         line = line.strip()
#         if not line:
#             continue  # Skip empty lines
        
#         if line.startswith('Model:'):
#             current_model = line.split('Model: ')[1].strip()
#             results[current_model] = {'validation': -1, 'ood_test': None}  # Initialize for the new model
#         else:
#             try:
#                 row = ast.literal_eval(line)
#             except (ValueError, SyntaxError):
#                 print(f"Skipping invalid line: {line}")
#                 continue
            
#             validation_score = row.get('validation', None)
#             ood_test_score = row.get('ood_test', None)
            
#             if validation_score is not None and ood_test_score is not None:
#                 if validation_score >= results[current_model]['validation']:
#                     results[current_model] = {'validation': validation_score, 'ood_test': ood_test_score}
#             elif ood_test_score is not None:
#                 results[current_model]['ood_test'] = ood_test_score
#     return results

# model_order = [
#     "dann", "irm", "ib_irm", "ib_erm", "vrex", "mixup",  # Red
#     "mmd", "causirl_mmd", "deepcoral", "causirl_coral",  # Blue
#     "aldro", "label_group_dro", "group_dro",  # Green
#     "expgrad", "ft_transformer", "lightgbm", "mlp", "node", "resnet", "saint", "tabtransformer", "xgb" # Pink
# ]

# model_name_mapping = {
#     "dann": "DANN", "irm": "IRM", "ib_irm": "IB-IRM", "ib_erm": "IB-ERM", "vrex": "VREX", "mixup": "MixUp",
#     "mmd": "MMD", "causirl_mmd": "Causirl_MMD", "deepcoral": "DeepCORAL", "causirl_coral": "Causirl_CORAL",
#     "aldro": "Adv. Label DRO", "label_group_dro": "Label GDRO", "group_dro": "GDRO",
#     "expgrad": "ExpGrad", "ft_transformer": "FT-Transformer", "lightgbm": "LightGBM", "mlp": "MLP", "node": "NODE",
#     "resnet": "ResNet", "saint": "SAINT", "tabtransformer": "TabTransformer", "xgb": "XGB"
# }

# color_mapping = {
#     "dann": "#009e74", "irm": "#009e74", "ib_irm": "#009e74", "ib_erm": "#009e74", "vrex": "#009e74", "mixup": "#009e74",
#     "mmd": "#d55e00", "causirl_mmd": "#d55e00", "deepcoral": "#d55e00", "causirl_coral": "#d55e00",
#     "aldro": "#de8f08", "label_group_dro": "#de8f08", "group_dro": "#de8f08",
#     "expgrad": "#0173b3", "ft_transformer": "#0173b3", "lightgbm": "#0173b3", "mlp": "#0173b3", "node": "#0173b3",
#     "resnet": "#0173b3", "saint": "#0173b3", "tabtransformer": "#0173b3", "xgb": "#0173b3",
# }

# ood_scores = {model: [] for model in model_order}
# for i in range(10):
#     file_path = f'results/synthetic_{i}.txt'
#     data = read_data_from_file(file_path)
#     results = process_data(data)
    
#     for model in model_order:
#         ood_scores[model].append(results[model]['ood_test'] if model in results and results[model]['ood_test'] is not None else 0)

# max_score = {i: max(ood_scores[model][i] for model in model_order) for i in range(10)}
# bar_scores = {model: [ood_scores[model][i] / max_score[i] for i in range(10)] for model in model_order}
# final_scores = {model: np.mean(bar_scores[model]) for model in model_order}
# std_scores = {model: np.std(bar_scores[model]) for model in model_order}

# sorted_models = sorted(final_scores, key=final_scores.get, reverse=True)
# sorted_scores = [final_scores[model] for model in sorted_models]
# sorted_std = [std_scores[model] for model in sorted_models]

# fig, ax = plt.subplots(figsize=(8, 10))
# y_pos = np.arange(len(sorted_models))
# ax.barh(y_pos, sorted_scores, xerr=sorted_std, color=[color_mapping[model] for model in sorted_models])
# ax.set_yticks(y_pos)
# ax.set_yticklabels(range(1, len(sorted_models) + 1), fontsize=16)
# ax.set_xticks(np.linspace(0, 1, num=6))  # Explicitly setting tick positions
# ax.set_xticklabels([f"{tick:.1f}" for tick in ax.get_xticks()], fontsize=16)
# ax.set_xlabel('Percentage of Max Accuracy (PMA-OOD)', fontsize=20)
# ax.set_ylabel('Model Rank', fontsize=20)
# ax.set_title('Percentage of Max Accuracy (PMA-OOD) on \n Synthetic Dataset Across All ratios ($\\rho$)', fontsize=20)
# ax.invert_yaxis()

# for i, v in enumerate(sorted_scores):
#     ax.text(v / 2, i, f'{model_name_mapping[sorted_models[i]]}', va='center', ha='center', color='white', fontweight='bold', fontsize=12)

# legend_labels = {
#     "#0173b3": "Baselines",
#     "#de8f08": "Robust Learning",
#     "#d55e00": "Domain Adaptation",
#     "#009e74": "Domain Generalization"
# }

# handles = [plt.Rectangle((0, 0), 1, 1, color=color) for color in legend_labels.keys()]
# ax.legend(handles, legend_labels.values(), loc='lower center', bbox_to_anchor=(0.5, -0.25), ncol=2, fontsize=16)

# plt.tight_layout()
# plt.savefig('oodtest_vertical_bar.png')
# plt.savefig('oodtest_vertical_bar.pdf')
# plt.show()

import matplotlib.pyplot as plt
import numpy as np
import ast

# Function to read data from a text file
def read_data_from_file(file_path):
    with open(file_path, 'r') as file:
        data = file.readlines()
    return data

# Function to process data and return results
def process_data(data):
    results = {}
    current_model = None
    
    for line in data:
        line = line.strip()
        if not line:
            continue  # Skip empty lines
        
        if line.startswith('Model:'):
            current_model = line.split('Model: ')[1].strip()
            results[current_model] = {'validation': -1, 'ood_test': None}  # Initialize for the new model
        else:
            try:
                row = ast.literal_eval(line)
            except (ValueError, SyntaxError):
                print(f"Skipping invalid line: {line}")
                continue
            
            validation_score = row.get('validation', None)
            ood_test_score = row.get('ood_test', None)
            
            if validation_score is not None and ood_test_score is not None:
                if validation_score >= results[current_model]['validation']:
                    results[current_model] = {'validation': validation_score, 'ood_test': ood_test_score}
            elif ood_test_score is not None:
                results[current_model]['ood_test'] = ood_test_score
    return results

model_order = [
    "dann", "irm", "ib_irm", "ib_erm", "vrex", "mixup",  # Red
    "mmd", "causirl_mmd", "deepcoral", "causirl_coral",  # Blue
    "aldro", "label_group_dro", "group_dro",  # Green
    "expgrad", "ft_transformer", "lightgbm", "mlp", "node", "resnet", "saint", "tabtransformer", "xgb" # Pink
]

model_name_mapping = {
    "dann": "DANN", "irm": "IRM", "ib_irm": "IB_IRM", "ib_erm": "IB_ERM", "vrex": "VREX", "mixup": "MixUp",
    "mmd": "MMD", "causirl_mmd": "Causirl_MMD", "deepcoral": "DeepCORAL", "causirl_coral": "Causirl_CORAL",
    "aldro": "Adv. Label DRO", "label_group_dro": "Label GDRO", "group_dro": "GDRO",
    "expgrad": "ExpGrad", "ft_transformer": "FT-Transformer", "lightgbm": "LightGBM", "mlp": "MLP", "node": "NODE",
    "resnet": "ResNet", "saint": "SAINT", "tabtransformer": "TabTransformer", "xgb": "XGB"
}

color_mapping = {
    "dann": "#009e74", "irm": "#009e74", "ib_irm": "#009e74", "ib_erm": "#009e74", "vrex": "#009e74", "mixup": "#009e74",
    "mmd": "#d55e00", "causirl_mmd": "#d55e00", "deepcoral": "#d55e00", "causirl_coral": "#d55e00",
    "aldro": "#de8f08", "label_group_dro": "#de8f08", "group_dro": "#de8f08",
    "expgrad": "#0173b3", "ft_transformer": "#0173b3", "lightgbm": "#0173b3", "mlp": "#0173b3", "node": "#0173b3",
    "resnet": "#0173b3", "saint": "#0173b3", "tabtransformer": "#0173b3", "xgb": "#0173b3",
}

def compute_scores(file_pattern):
    ood_scores = {model: [] for model in model_order}
    for i in range(10):
        file_path = file_pattern.format(i)
        data = read_data_from_file(file_path)
        results = process_data(data)
        
        for model in model_order:
            ood_scores[model].append(results[model]['ood_test'] if model in results and results[model]['ood_test'] is not None else 0)

    max_score = {i: max(ood_scores[model][i] for model in model_order) for i in range(10)}
    bar_scores = {model: [ood_scores[model][i] / max_score[i] for i in range(10)] for model in model_order}
    final_scores = {model: np.mean(bar_scores[model]) for model in model_order}
    std_scores = {model: np.std(bar_scores[model]) for model in model_order}

    return final_scores, std_scores

final_scores_1, std_scores1 = compute_scores('results/synthetic_{}.txt')
final_scores_2, std_scores2 = compute_scores('results/synthetic_{}_nonlinear.txt')

fig, axes = plt.subplots(1,2, figsize=(16, 8))

for i, (ax, final_scores, std_scores) in enumerate(zip(axes, [final_scores_1, final_scores_2],[std_scores1, std_scores2])):
    sorted_models = sorted(final_scores, key=final_scores.get, reverse=True)
    sorted_scores = [final_scores[model] for model in sorted_models]
    sorted_stds = [std_scores[model] for model in sorted_models]

    y_pos = np.arange(len(sorted_models))
    ax.barh(y_pos, sorted_scores, xerr=sorted_stds, color=[color_mapping[model] for model in sorted_models])
    ax.set_yticks(y_pos)
    ax.set_yticklabels(range(1, len(sorted_models) + 1), fontsize=16)
    ax.set_xticks(np.linspace(0, 1, num=6))  # Set x-axis limit to 1 and adjust ticks
    ax.set_xticklabels([f"{tick:.1f}" for tick in ax.get_xticks()], fontsize=16)
    ax.set_xlabel('Percentage of Max Accuracy (PMA-OOD)', fontsize=18)
    if i==0:
        ax.set_ylabel('Model Rank', fontsize=18)

    ax.invert_yaxis()
    for i, v in enumerate(sorted_scores):
        ax.text(v / 2, i, f'{model_name_mapping[sorted_models[i]]}', va='center', ha='center', color='white', fontweight='bold', fontsize=12)

legend_labels = {
    "#0173b3": "ERM-Based Methods",
    "#de8f08": "Robust Learning Methods",
    "#d55e00": "Domain Adaptation Methods",
    "#009e74": "Domain Generalization Methods"
}
fig.suptitle('Percentage of Max Accuracy (PMA-OOD) on Synthetic Datasets Across All ratios ($\\rho=\\frac{|\mathbf{X}_c|}{|\mathbf{X}_a|}$)\n '
'Left: Data With Linear Equations, Right: Data With Nonlinear Equations ', fontsize=20)

handles = [plt.Rectangle((0, 0), 1, 1, color=color) for color in legend_labels.keys()]
fig.legend(handles, legend_labels.values(), loc='lower center', bbox_to_anchor=(0.5, -0.075), ncol=4, fontsize=16)

plt.tight_layout()
plt.savefig('pmaood.png',bbox_inches='tight')
plt.savefig('pmaood.pdf',bbox_inches='tight')
plt.show()
