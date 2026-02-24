import ast
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def read_data_from_file(file_path):
    with open(file_path, 'r') as file:
        data = file.readlines()
    return data

# Function to process data and return results per model
def process_data(data):
    results = {}
    current_model = None

    for line in data:
        line = line.strip()
        if not line:
            continue  # Skip empty lines
        
        # If the line specifies a model, initialize its entry
        if line.startswith('Model:'):
            current_model = line.split('Model: ')[1].strip()
            results[current_model] = {'validation': -1, 'ood_test': None}
        else:
            # Convert the string representation of the dictionary to an actual dictionary
            try:
                row = ast.literal_eval(line)
            except (ValueError, SyntaxError):
                print(f"Skipping invalid line: {line}")
                continue
            
            validation_score = row.get('validation', None)
            ood_test_score = row.get('ood_test', None)
            
            # Update the model result if the validation score is better
            if validation_score is not None and ood_test_score is not None:
                if validation_score >= results[current_model]['validation']:
                    results[current_model] = {'validation': validation_score, 'ood_test': ood_test_score}
            elif ood_test_score is not None:
                results[current_model]['ood_test'] = ood_test_score
    return results

# Define families with their corresponding model name substrings
families = {
    "Domain Generalization": ["dann", "irm", "ib_irm", "ib_erm", "vrex", "mixup"],
    "Domain Adaptation": ["mmd", "causirl_mmd", "deepcoral", "causirl_coral"],
    "Robust Learning": ["aldro", "label_group_dro", "group_dro"],
    "ERM-Based": ["expgrad", "ft_transformer", "lightgbm", "mlp", "node", "resnet", "saint", "tabtransformer", "xgb"]
}

# Settings for which we want to plot results
settings = [1,2]

# Data structure to store average OOD test accuracy per family per setting
family_avg_ood = {family: [] for family in families.keys()}

for setting in settings:
    # Initialize a dictionary to accumulate scores for each family for the current setting
    family_scores = {family: [] for family in families.keys()}
    for seed in range(1, 4):
        file_name = f"results/informative/setting{setting}_seed{seed}_nonlinear.txt"
        try:
            data = read_data_from_file(file_name)
            results = process_data(data)
            # For each model in the file, collect its ood_test score if available
            for model, metrics in results.items():
                ood_test = metrics.get('ood_test')
                if ood_test is not None:
                    model_lower = model.lower()  # convert to lower-case for matching
                    # Check which family the model belongs to
                    for family, methods in families.items():
                        if any(method in model_lower for method in methods):
                            family_scores[family].append(ood_test)
                            break  # Stop checking once the model is assigned to a family
        except Exception as e:
            print(e)
            print(f"Error processing file {file_name}")
    # Compute the average ood_test for each family for the current setting
    for family in families.keys():
        scores = family_scores[family]
        if scores:
            avg_score = sum(scores) / len(scores)
        else:
            avg_score = None
        family_avg_ood[family].append(avg_score)

# Plot the average OOD test accuracy for each family across settings
plt.figure(figsize=(10, 6))
for family, avg_scores in family_avg_ood.items():
    plt.plot(settings, avg_scores, marker='o', label=family)
plt.xlabel('Setting')
plt.ylabel('Average OOD Test Accuracy')
plt.title('Average OOD Test Accuracy by Model Family Across Settings')
plt.legend()
plt.grid(True)
plt.savefig('average_ood_test_by_family_across_settings.png')
