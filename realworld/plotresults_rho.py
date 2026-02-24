import matplotlib.pyplot as plt
import ast
from matplotlib.patches import Patch

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
        
        # Check if the line specifies a model
        if line.startswith('Model:'):
            current_model = line.split('Model: ')[1].strip()
            results[current_model] = {'validation': -1, 'ood_test': None}  # Initialize for the new model
        else:
            # Convert the string representation of the dictionary to an actual dictionary
            try:
                row = ast.literal_eval(line)
            except (ValueError, SyntaxError):
                print(f"Skipping invalid line: {line}")
                continue
            
            # Extract validation and ood_test scores
            validation_score = row.get('validation', None)
            ood_test_score = row.get('ood_test', None)
            
            if validation_score is not None and ood_test_score is not None:
                # Update if the current validation score is better
                if validation_score >= results[current_model]['validation']:
                    results[current_model] = {'validation': validation_score, 'ood_test': ood_test_score}
            elif ood_test_score is not None:
                # Handle rows with only ood_test scores
                results[current_model]['ood_test'] = ood_test_score
    return results

# Define the order of models and their corresponding colors
model_order = [
    "dann", "irm", "ib_irm", "ib_erm", "vrex", "mixup",  # Red
    "mmd", "causirl_mmd", "deepcoral", "causirl_coral",  # Blue
    "aldro", "label_group_dro", "group_dro",  # Green
    "expgrad", "ft_transformer", "lightgbm", "mlp", "node", "resnet", "saint", "tabtransformer", "xgb" # Pink
]

# Define the color mapping
color_mapping = {
    "dann": "#009e74",
    "irm": "#009e74",
    "ib_irm": "#009e74",
    "ib_erm": "#009e74",
    "vrex": "#009e74",
    "mixup": "#009e74",
    "mmd": "#d55e00",
    "causirl_mmd": "#d55e00",
    "deepcoral": "#d55e00",
    "causirl_coral": "#d55e00",
    "aldro": "#de8f08",
    "label_group_dro": "#de8f08",
    "group_dro": "#de8f08",
    "expgrad": "#0173b3",
    "ft_transformer": "#0173b3",
    "lightgbm": "#0173b3",
    "mlp": "#0173b3",
    "node": "#0173b3",
    "resnet": "#0173b3",
    "saint": "#0173b3",
    "tabtransformer": "#0173b3",
    "xgb": "#0173b3",
}
model_name_mapping = {
    "dann": "DANN", "irm": "IRM", "ib_irm": "IB_IRM", "ib_erm": "IB_ERM", "vrex": "VREX", "mixup": "MixUp",
    "mmd": "MMD", "causirl_mmd": "Causirl_MMD", "deepcoral": "DeepCORAL", "causirl_coral": "Causirl_CORAL",
    "aldro": "Adv. Label DRO", "label_group_dro": "Label GDRO", "group_dro": "GDRO",
    "expgrad": "ExpGrad", "ft_transformer": "FT-Transformer", "lightgbm": "LightGBM", "mlp": "MLP", "node": "NODE",
    "resnet": "ResNet", "saint": "SAINT", "tabtransformer": "TabTransformer", "xgb": "XGB"
}
# Create a figure with 10 subplots
fig, axes = plt.subplots(2, 6, figsize=(25, 12))  # 2 rows, 5 columns
fig.suptitle('OOD Test Accuracy by Model On Linear Synthetic Dataset', fontsize=24)

# Loop through 10 files
for i in range(0, 11):
    file_path = f'results/synthetic_{i}.txt'  # File path for each result file
    data = read_data_from_file(file_path)
    results = process_data(data)
    
    # Prepare data for plotting
    model_names = model_order  # Always use the full model order
    ood_scores = []
    colors = []
    
    for model in model_order:
        if model in results and results[model]['ood_test'] is not None:
            ood_scores.append(results[model]['ood_test'])
        else:
            ood_scores.append(0)  # Use 0 for missing methods
        colors.append(color_mapping[model])
    
    # Plot in the corresponding subplot
    ax = axes[i // 6, i % 6]  # Determine subplot position
    bars = ax.bar(model_names, ood_scores, color=colors)
    ax.set_xticks(range(len(model_names)))
    ax.set_xticklabels([model_name_mapping[model] for model in model_names], rotation=45, ha='right')
    if i==0 or i==6:
        ax.set_ylabel('OOD Test Accuracy',fontsize=20)
    ax.set_title(f'$\\rho=0.{i}$',fontsize=18)
    
    # Add a horizontal line at the highest accuracy
    max_ood_score = max(ood_scores) if ood_scores else 0
    ax.axhline(max_ood_score, color='gray', linestyle='--', linewidth=1)
    # ax.legend(loc='upper right')

fig.delaxes(axes[1,5])
# Add a legend for the colors

legend_elements = [
    Patch(facecolor='#009e74', label='Domain Generalization Methods'),
    Patch(facecolor='#d55e00', label='Domain Adaptation Methods'),
    Patch(facecolor='#de8f08', label='Robust Learning Methods'),
    Patch(facecolor='#0173b3', label='ERM-Based Methods')
]
fig.legend(handles=legend_elements, bbox_to_anchor=(0.5, -0.075), loc='lower center', ncol=4, fontsize=22)

# Adjust layout and save the figure
plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust layout to make room for the title
plt.savefig('rho_linear.pdf',bbox_inches='tight')
plt.savefig('rho_linear.png',bbox_inches='tight')
plt.show()