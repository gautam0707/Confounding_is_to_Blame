import ast
import matplotlib.pyplot as plt

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

# Compute average ood_test for each setting (i=1,...,8) across seeds (j=1,...,3)
settings = [1,2]#list(range(1, 3))
avg_ood_test = []

for setting in settings:
    setting_scores = []
    for seed in range(1, 4):
        file_name = f"results/informative/setting{setting}_seed{seed}_linear.txt"
        try:
            data = read_data_from_file(file_name)
            results = process_data(data)
            # For each model in the file, collect its ood_test score if available
            for model, metrics in results.items():
                ood_test = metrics.get('ood_test')
                if ood_test is not None:
                    setting_scores.append(ood_test)
        except Exception as e:
            print(e)
            print(f"Error processing file {file_name}")
    # Compute the average ood_test for the current setting if any scores were found
    if setting_scores:
        avg = sum(setting_scores) / len(setting_scores)
    else:
        avg = None
    avg_ood_test.append(avg)

# Plot the average ood_test accuracy for each setting
plt.figure(figsize=(10, 6))
plt.plot(settings, avg_ood_test, marker='o')
plt.xlabel('Setting')
plt.ylabel('Average OOD Test Accuracy')
plt.title('Average OOD Test Accuracy Across Settings')
plt.grid(True)
plt.savefig('average_ood_test_across_settings.png')
