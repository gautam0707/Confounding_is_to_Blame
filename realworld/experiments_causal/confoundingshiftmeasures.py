# import argparse
# import numpy as np
# import pandas as pd
# import torch
# import torch.nn as nn
# import torch.optim as optim
# import torch.nn.functional as F
# from torch.utils.data import TensorDataset, DataLoader
# import xgboost as xgb
# from sklearn.decomposition import PCA
# from sklearn.preprocessing import StandardScaler
# from sklearn.pipeline import Pipeline
# from tableshift import get_dataset
# import npeet.entropy_estimators as ee
# import os
# import json
# import traceback

# # Configure paths and parameters
# os.makedirs("results", exist_ok=True)
# OUTPUT_FILE = os.path.join("results", "aggregated_results.txt")
# DEFAULT_PCA_COMPONENTS = 50

# # Neural network model with integrated dimensionality reduction
# class MLP(nn.Module):
#     def __init__(self, input_dim, hidden_dim=64):
#         super().__init__()
#         self.fc1 = nn.Linear(input_dim, hidden_dim)
#         self.fc2 = nn.Linear(hidden_dim, 1)
        
#     def forward(self, x):
#         x = F.relu(self.fc1(x))
#         return torch.sigmoid(self.fc2(x))
    
#     def get_representation(self, x):
#         return F.relu(self.fc1(x))

# def append_to_file(text):
#     with open(OUTPUT_FILE, "a") as f:
#         f.write(text + "\n")

# def compute_mi_terms(rep, hatY, Y_vals, E):
#     return (
#         ee.cmi(rep, Y_vals, E),  # inf
#         ee.cmi(rep, E, Y_vals),  # inv
#         ee.mi(rep, E),           # lcs
#         ee.cmi(Y_vals, E, rep),  # cs
#         ee.cmi(rep, Y_vals, hatY) # res
#     )

# def create_preprocessor(pca_components):
#     return Pipeline([
#         ('scaler', StandardScaler()),
#         ('pca', PCA(n_components=pca_components))
#     ])

# def train_pytorch_model(X_train, y_train, domains_train, model_type, device, pca_components, epochs=50):
#     preprocessor = create_preprocessor(pca_components)
#     X_processed = preprocessor.fit_transform(X_train)
    
#     # Convert to tensors
#     X_tensor = torch.tensor(X_processed, dtype=torch.float32).to(device)
#     y_tensor = torch.tensor(y_train.values, dtype=torch.float32).to(device)
#     e_tensor = torch.tensor(domains_train.values, dtype=torch.long).to(device)
    
#     # Create model with reduced input size
#     model = MLP(pca_components).to(device)
#     optimizer = optim.Adam(model.parameters(), lr=0.001)
    
#     # Training loop
#     dataset = TensorDataset(X_tensor, y_tensor, e_tensor)
#     loader = DataLoader(dataset, batch_size=64, shuffle=True)
    
#     for epoch in range(epochs):
#         for x, y, e in loader:
#             optimizer.zero_grad()
#             outputs = model(x).squeeze()
            
#             # Loss calculation based on model type
#             if model_type == "groupdro":
#                 losses = F.binary_cross_entropy(outputs, y, reduction='none')
#                 group_losses = [losses[e == g].mean() for g in torch.unique(e)]
#                 group_losses = torch.stack([gl for gl in group_losses if not torch.isnan(gl)])
#                 weights = torch.ones_like(group_losses) / len(group_losses)
#                 loss = torch.sum(weights * group_losses)
                
#             elif model_type == "vrex":
#                 losses = F.binary_cross_entropy(outputs, y, reduction='none')
#                 domain_losses = [losses[e == d].mean() for d in torch.unique(e)]
#                 if len(domain_losses) >= 2:
#                     domain_losses = torch.stack(domain_losses)
#                     loss = domain_losses.mean() + 0.1 * domain_losses.var()
#                 else:
#                     loss = losses.mean()
#             else:
#                 loss = F.binary_cross_entropy(outputs, y)
            
#             loss.backward()
#             optimizer.step()
    
#     # Get final representations
#     with torch.no_grad():
#         reps = model.get_representation(X_tensor).cpu().numpy()
#         hatY = (model(X_tensor).squeeze().cpu().numpy() > 0.5).astype(int)
    
#     return reps, hatY, preprocessor

# def extract_representations_and_compute_mi(X_train, X_test, y_train, y_test, domain_train, domain_test, model_type, device, pca_components):
#     # Combine data for representation learning
#     X_combined = pd.concat([X_train, X_test])
#     y_combined = np.concatenate([y_train, y_test])
#     domain_combined = np.concatenate([domain_train, domain_test])
    
#     preprocessor = create_preprocessor(pca_components)
#     X_processed = preprocessor.fit_transform(X_combined)
    
#     if model_type == "xgboost":
#         model = xgb.XGBClassifier()
#         model.fit(X_processed, y_combined)
#         hatY = model.predict(X_processed)
#         rep = X_processed
#     else:
#         reps, hatY, _ = train_pytorch_model(
#             X_combined, pd.Series(y_combined),
#             pd.Series(domain_combined), model_type,
#             device, pca_components
#         )
#         rep = reps
    
#     return compute_mi_terms(rep, hatY, y_combined, domain_combined)

# def compute_model_accuracy(model_type, X_train, X_test, y_train, y_test, device, pca_components):
#     preprocessor = create_preprocessor(pca_components)
#     X_train_processed = preprocessor.fit_transform(X_train)
#     X_test_processed = preprocessor.transform(X_test)
    
#     if model_type == "xgboost":
#         model = xgb.XGBClassifier()
#         model.fit(X_train_processed, y_train)
#         return (
#             model.score(X_train_processed, y_train),
#             model.score(X_test_processed, y_test)
#         )
#     else:
#         # Initialize model with PCA dimensions
#         model = MLP(pca_components).to(device)
#         optimizer = optim.Adam(model.parameters())
        
#         # Convert data to tensors
#         X_tensor = torch.tensor(X_train_processed, dtype=torch.float32).to(device)
#         y_tensor = torch.tensor(y_train.values, dtype=torch.float32).to(device)
        
#         # Training loop
#         for _ in range(50):
#             optimizer.zero_grad()
#             outputs = model(X_tensor).squeeze()
#             loss = F.binary_cross_entropy(outputs, y_tensor)
#             loss.backward()
#             optimizer.step()
        
#         # Calculate accuracies
#         with torch.no_grad():
#             train_preds = (model(X_tensor).squeeze().cpu().numpy() > 0.5)
#             id_acc = (train_preds == y_train.values).mean()
            
#             X_test_tensor = torch.tensor(X_test_processed, dtype=torch.float32).to(device)
#             test_preds = (model(X_test_tensor).squeeze().cpu().numpy() > 0.5)
#             ood_acc = (test_preds == y_test.values).mean()
        
#         return id_acc, ood_acc

# def main(experiment, cache_dir, model_type, pca_components, debug=False):
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     results = {}
    
#     for setting in ["causal", "arguablycausal", "all"]:
#         if setting not in experiment:
#             continue
            
#         # Load dataset
#         dataset = get_dataset(experiment[setting], cache_dir)
#         X_train, y_train, _, domain_train = dataset.get_pandas("train")
#         X_test, y_test, _, domain_test = dataset.get_pandas("ood_test")
        
#         # Compute MI terms with dimensionality reduction
#         mi_terms = extract_representations_and_compute_mi(
#             X_train, X_test, y_train, y_test,
#             domain_train, domain_test, model_type,
#             device, pca_components
#         )
        
#         # Calculate model accuracies
#         id_acc, ood_acc = compute_model_accuracy(
#             model_type, X_train, X_test,
#             y_train, y_test, device, pca_components
#         )
        
#         # Store results
#         results[setting] = {
#             "Covariate_Shift": float(mi_terms[2]),
#             "Concept_Shift": float(mi_terms[3]),
#             "ID_Accuracy": float(id_acc),
#             "OOD_Accuracy": float(ood_acc),
#             "PCA_Components": pca_components
#         }
        
#         # Write to output file
#         summary = (
#             f"Model: {model_type} | Dataset: {experiment['all']} | Setting: {setting}\n"
#             f"  Covariate Shift: {mi_terms[2]:.4f}\n"
#             f"  Concept Shift: {mi_terms[3]:.4f}\n"
#             f"  ID Accuracy: {id_acc:.4f}\n"
#             f"  OOD Accuracy: {ood_acc:.4f}\n"
#             "----------------------------------------"
#         )
#         append_to_file(summary)
    
#     # Save results to JSON
#     output_file = os.path.join("results", f"{model_type}_{experiment['all']}_results.json")
#     with open(output_file, "w") as f:
#         json.dump(results, f, indent=4)
    
#     return results

# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(description="Domain Shift Analysis with Dimensionality Reduction")
#     parser.add_argument("--cache_dir", default="tmp", help="Dataset cache directory")
#     parser.add_argument("--model", required=True, choices=["mlp", "xgboost", "groupdro", "vrex"])
#     parser.add_argument("--pca_components", type=int, default=DEFAULT_PCA_COMPONENTS,
#                        help="Number of PCA components for dimensionality reduction")
#     parser.add_argument("--debug", action="store_true", help="Enable debug mode")
#     args = parser.parse_args()

#     experiments = [
#         {"causal": "diabetes_readmission_causal", "arguablycausal": "diabetes_readmission_arguablycausal", "all": "diabetes_readmission"},
#         {"causal": "acsfoodstamps_causal", "arguablycausal": "acsfoodstamps_arguablycausal", "all": "acsfoodstamps"},
#         {"causal": "acsincome_causal", "arguablycausal": "acsincome_arguablycausal", "anticausal": "acsincome_anticausal", "all": "acsincome"},
#         {"causal": "acspubcov_causal", "arguablycausal": "acspubcov_arguablycausal", "all": "acspubcov"},
#         {"causal": "acsunemployment_causal", "arguablycausal": "acsunemployment_arguablycausal", "anticausal": "acsunemployment_anticausal", "all": "acsunemployment"},
#         {"causal": "brfss_diabetes_causal", "arguablycausal": "brfss_diabetes_arguablycausal", "anticausal": "brfss_diabetes_anticausal", "all": "brfss_diabetes"},
#         {"causal": "brfss_blood_pressure_causal", "arguablycausal": "brfss_blood_pressure_arguablycausal", "anticausal": "brfss_blood_pressure_anticausal", "all": "brfss_blood_pressure"},
#         {"causal": "college_scorecard_causal", "arguablycausal": "college_scorecard_arguablycausal", "all": "college_scorecard"},
#         {"causal": "assistments_causal", "arguablycausal": "assistments_arguablycausal", "all": "assistments"},
#     ]

#     for exp in experiments:
#         try:
#             print(f"Processing experiment: {exp['all']}")
#             main(exp, args.cache_dir, args.model, args.pca_components, args.debug)
#         except Exception as e:
#             print(f"Error processing {exp['all']}: {str(e)}")
#             print(traceback.format_exc())


# import argparse
# import numpy as np
# import pandas as pd
# import torch
# import torch.nn as nn
# import torch.optim as optim
# import torch.nn.functional as F
# from torch.utils.data import TensorDataset, DataLoader
# import xgboost as xgb
# from sklearn.decomposition import PCA
# from sklearn.preprocessing import StandardScaler
# from sklearn.pipeline import Pipeline
# from tableshift import get_dataset
# import npeet.entropy_estimators as ee
# import os
# import json
# import traceback

# # Configure paths and parameters
# os.makedirs("results", exist_ok=True)
# OUTPUT_FILE = os.path.join("results", "aggregated_results.txt")
# DEFAULT_PCA_COMPONENTS = 50

# # Neural network model
# class MLP(nn.Module):
#     def __init__(self, input_dim, hidden_dim=64):
#         super().__init__()
#         self.fc1 = nn.Linear(input_dim, hidden_dim)
#         self.fc2 = nn.Linear(hidden_dim, 1)
        
#     def forward(self, x):
#         x = F.relu(self.fc1(x))
#         return torch.sigmoid(self.fc2(x))
    
#     def get_representation(self, x):
#         return F.relu(self.fc1(x))

# def append_to_file(text):
#     with open(OUTPUT_FILE, "a") as f:
#         f.write(text + "\n")

# def compute_mi_terms(rep, hatY, Y_vals, E):
#     return (
#         ee.cmi(rep, Y_vals, E),  # inf
#         ee.cmi(rep, E, Y_vals),  # inv
#         ee.mi(rep, E),           # lcs
#         ee.cmi(Y_vals, E, rep),  # cs
#         ee.cmi(rep, Y_vals, hatY) # res
#     )

# def create_preprocessor(pca_components):
#     return Pipeline([
#         ('scaler', StandardScaler()),
#         ('pca', PCA(n_components=pca_components))
#     ])

# def train_pytorch_model(X_train, y_train, domains_train, model_type, device, pca_components, epochs=50):
#     preprocessor = create_preprocessor(pca_components)
#     X_processed = preprocessor.fit_transform(X_train)
    
#     # Create CPU tensors first
#     X_tensor = torch.tensor(X_processed, dtype=torch.float32)
#     y_tensor = torch.tensor(y_train.values, dtype=torch.float32)
#     e_tensor = torch.tensor(domains_train.values, dtype=torch.long)
    
#     model = MLP(pca_components).to(device)
#     optimizer = optim.Adam(model.parameters(), lr=0.001)
    
#     dataset = TensorDataset(X_tensor, y_tensor, e_tensor)
#     loader = DataLoader(
#         dataset,
#         batch_size=1024,
#         shuffle=True,
#         num_workers=4,
#         pin_memory=torch.cuda.is_available(),
#         persistent_workers=False
#     )
    
#     for epoch in range(epochs):
#         model.train()
#         for x, y, e in loader:
#             # Move data to device inside training loop
#             x = x.to(device, non_blocking=True)
#             y = y.to(device, non_blocking=True)
#             e = e.to(device, non_blocking=True)
            
#             optimizer.zero_grad(set_to_none=True)
#             outputs = model(x).squeeze()
            
#             if model_type == "groupdro":
#                 losses = F.binary_cross_entropy(outputs, y, reduction='none')
#                 group_losses = [losses[e == g].mean() for g in torch.unique(e)]
#                 group_losses = torch.stack([gl for gl in group_losses if not torch.isnan(gl)])
#                 weights = torch.ones_like(group_losses) / len(group_losses)
#                 loss = torch.sum(weights * group_losses)
                
#             elif model_type == "vrex":
#                 losses = F.binary_cross_entropy(outputs, y, reduction='none')
#                 domain_losses = [losses[e == d].mean() for d in torch.unique(e)]
#                 if len(domain_losses) >= 2:
#                     domain_losses = torch.stack(domain_losses)
#                     loss = domain_losses.mean() + 0.1 * domain_losses.var()
#                 else:
#                     loss = losses.mean()
#             else:
#                 loss = F.binary_cross_entropy(outputs, y)
            
#             loss.backward()
#             optimizer.step()
    
#     # Get final representations on CPU
#     with torch.no_grad():
#         model.eval()
#         X_full = X_tensor.to(device)
#         reps = model.get_representation(X_full).cpu().numpy()
#         hatY = (model(X_full).squeeze().cpu().numpy() > 0.5).astype(int)
    
#     return reps, hatY, preprocessor

# def extract_representations_and_compute_mi(X_train, X_test, y_train, y_test, domain_train, domain_test, model_type, device, pca_components):
#     X_combined = pd.concat([X_train, X_test])
#     y_combined = np.concatenate([y_train, y_test])
#     domain_combined = np.concatenate([domain_train, domain_test])
    
#     preprocessor = create_preprocessor(pca_components)
#     X_processed = preprocessor.fit_transform(X_combined)
    
#     if model_type == "xgboost":
#         model = xgb.XGBClassifier(
#             tree_method='gpu_hist' if torch.cuda.is_available() else 'hist',
#             predictor='gpu_predictor' if torch.cuda.is_available() else 'cpu_predictor',
#             n_jobs=-1
#         )
#         model.fit(X_processed, y_combined)
#         hatY = model.predict(X_processed)
#         rep = X_processed
#     else:
#         reps, hatY, _ = train_pytorch_model(
#             X_combined, pd.Series(y_combined),
#             pd.Series(domain_combined), model_type,
#             device, pca_components
#         )
#         rep = reps
    
#     return compute_mi_terms(rep, hatY, y_combined, domain_combined)

# def compute_model_accuracy(model_type, X_train, X_test, y_train, y_test, device, pca_components):
#     preprocessor = create_preprocessor(pca_components)
#     X_train_processed = preprocessor.fit_transform(X_train)
#     X_test_processed = preprocessor.transform(X_test)
    
#     if model_type == "xgboost":
#         model = xgb.XGBClassifier(
#             tree_method='gpu_hist' if torch.cuda.is_available() else 'hist',
#             n_jobs=-1
#         )
#         model.fit(X_train_processed, y_train)
#         return (
#             model.score(X_train_processed, y_train),
#             model.score(X_test_processed, y_test)
#         )
#     else:
#         model = MLP(pca_components).to(device)
#         optimizer = optim.Adam(model.parameters())
        
#         X_tensor = torch.tensor(X_train_processed, dtype=torch.float32)
#         y_tensor = torch.tensor(y_train.values, dtype=torch.float32)
        
#         train_dataset = TensorDataset(X_tensor, y_tensor)
#         train_loader = DataLoader(
#             train_dataset,
#             batch_size=1024,
#             shuffle=True,
#             pin_memory=torch.cuda.is_available()
#         )
        
#         for _ in range(50):
#             model.train()
#             for x, y in train_loader:
#                 x = x.to(device, non_blocking=True)
#                 y = y.to(device, non_blocking=True)
                
#                 optimizer.zero_grad()
#                 outputs = model(x).squeeze()
#                 loss = F.binary_cross_entropy(outputs, y)
#                 loss.backward()
#                 optimizer.step()
        
#         model.eval()
#         with torch.no_grad():
#             X_test_tensor = torch.tensor(X_test_processed, dtype=torch.float32)
#             test_preds = (model(X_test_tensor.to(device)).squeeze().cpu().numpy() > 0.5)
#             ood_acc = (test_preds == y_test.values).mean()
            
#             train_preds = (model(X_tensor.to(device)).squeeze().cpu().numpy() > 0.5)
#             id_acc = (train_preds == y_train.values).mean()
        
#         return id_acc, ood_acc

# def main(experiment, cache_dir, model_type, pca_components, debug=False):
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     results = {}
    
#     for setting in ["causal", "arguablycausal", "all"]:
#         if setting not in experiment:
#             continue
            
#         dataset = get_dataset(experiment[setting], cache_dir)
#         X_train, y_train, _, domain_train = dataset.get_pandas("train")
#         X_test, y_test, _, domain_test = dataset.get_pandas("ood_test")
        
#         mi_terms = extract_representations_and_compute_mi(
#             X_train, X_test, y_train, y_test,
#             domain_train, domain_test, model_type,
#             device, pca_components
#         )
        
#         id_acc, ood_acc = compute_model_accuracy(
#             model_type, X_train, X_test,
#             y_train, y_test, device, pca_components
#         )
        
#         results[setting] = {
#             "Informativeness": float(mi_terms[0]),
#             "Invariance": float(mi_terms[1]),
#             "Covariate_Shift": float(mi_terms[2]),
#             "Concept_Shift": float(mi_terms[3]),
#             "Residual": float(mi_terms[4]),
#             "ID_Accuracy": float(id_acc),
#             "OOD_Accuracy": float(ood_acc),
#             "PCA_Components": pca_components
#         }
        
#         summary = (
#             f"Model: {model_type} | Dataset: {experiment['all']} | Setting: {setting}\n"
#             f"  Informativeness: {mi_terms[0]:.4f}\n"
#             f"  Invariance: {mi_terms[1]:.4f}\n"
#             f"  Covariate Shift: {mi_terms[2]:.4f}\n"
#             f"  Concept Shift: {mi_terms[3]:.4f}\n"
#             f"  Residual: {mi_terms[4]:.4f}\n"
#             f"  ID Accuracy: {id_acc:.4f}\n"
#             f"  OOD Accuracy: {ood_acc:.4f}\n"
#             "----------------------------------------"
#         )
#         append_to_file(summary)
    
#     output_file = os.path.join("results", f"{model_type}_{experiment['all']}_results.json")
#     with open(output_file, "w") as f:
#         json.dump(results, f, indent=4)
    
#     return results

# if __name__ == "__main__":
#     torch.multiprocessing.set_start_method('spawn', force=True)
#     parser = argparse.ArgumentParser(description="Domain Shift Analysis")
#     parser.add_argument("--cache_dir", default="tmp")
#     parser.add_argument("--model", required=True, choices=["mlp", "xgboost", "groupdro", "vrex"])
#     parser.add_argument("--pca_components", type=int, default=DEFAULT_PCA_COMPONENTS)
#     parser.add_argument("--debug", action="store_true")
#     args = parser.parse_args()

#     experiments = [
#         {"causal": "diabetes_readmission_causal", "arguablycausal": "diabetes_readmission_arguablycausal", "all": "diabetes_readmission"},
#         {"causal": "acsfoodstamps_causal", "arguablycausal": "acsfoodstamps_arguablycausal", "all": "acsfoodstamps"},
#         {"causal": "acsincome_causal", "arguablycausal": "acsincome_arguablycausal", "anticausal": "acsincome_anticausal", "all": "acsincome"},
#         {"causal": "acspubcov_causal", "arguablycausal": "acspubcov_arguablycausal", "all": "acspubcov"},
#         {"causal": "acsunemployment_causal", "arguablycausal": "acsunemployment_arguablycausal", "anticausal": "acsunemployment_anticausal", "all": "acsunemployment"},
#         {"causal": "brfss_diabetes_causal", "arguablycausal": "brfss_diabetes_arguablycausal", "anticausal": "brfss_diabetes_anticausal", "all": "brfss_diabetes"},
#         {"causal": "brfss_blood_pressure_causal", "arguablycausal": "brfss_blood_pressure_arguablycausal", "anticausal": "brfss_blood_pressure_anticausal", "all": "brfss_blood_pressure"},
#         {"causal": "college_scorecard_causal", "arguablycausal": "college_scorecard_arguablycausal", "all": "college_scorecard"},
#         {"causal": "assistments_causal", "arguablycausal": "assistments_arguablycausal", "all": "assistments"},
#     ]

#     for exp in experiments:
#         try:
#             print(f"Processing experiment: {exp['all']}")
#             main(exp, args.cache_dir, args.model, args.pca_components, args.debug)
#         except Exception as e:
#             print(f"Error processing {exp['all']}: {str(e)}")
#             print(traceback.format_exc())
#         finally:
#             # Clean up CUDA memory
#             if torch.cuda.is_available():
#                 torch.cuda.empty_cache()

import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import xgboost as xgb
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from tableshift import get_dataset
import npeet.entropy_estimators as ee
import os
import json
import traceback

# Configure paths and parameters
os.makedirs("results", exist_ok=True)
OUTPUT_FILE = os.path.join("results", "aggregated_results.txt")
DEFAULT_PCA_COMPONENTS = 50

# Neural network model
class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim=64):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        return torch.sigmoid(self.fc2(x))
    
    def get_representation(self, x):
        return F.relu(self.fc1(x))

# Optionally compile the model for faster execution (PyTorch 2.0+)
def compile_model(model):
    try:
        return torch.compile(model)
    except Exception:
        return model

def append_to_file(text, filename=OUTPUT_FILE):
    with open(filename, "a") as f:
        f.write(text + "\n")

def compute_mi_terms(rep, hatY, Y_vals, E):
    return (
        ee.cmi(rep, Y_vals, E),  # inf
        ee.cmi(rep, E, Y_vals),  # inv
        ee.mi(rep, E),           # lcs
        ee.cmi(Y_vals, E, rep),  # cs
        ee.cmi(rep, Y_vals, hatY) # res
    )

def create_preprocessor(pca_components):
    return Pipeline([
        ('scaler', StandardScaler()),
        ('pca', PCA(n_components=pca_components))
    ])

def train_pytorch_model(X_train, y_train, domains_train, model_type, device, pca_components, epochs=50):
    # Preprocess data once
    preprocessor = create_preprocessor(pca_components)
    X_processed = preprocessor.fit_transform(X_train)
    
    # Convert data to tensors
    X_tensor = torch.tensor(X_processed, dtype=torch.float32)
    y_tensor = torch.tensor(y_train.values, dtype=torch.float32)
    e_tensor = torch.tensor(domains_train.values, dtype=torch.long)
    
    model = MLP(pca_components).to(device)
    # Optionally compile model for faster execution (PyTorch 2.0+)
    model = compile_model(model)
    
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    dataset = TensorDataset(X_tensor, y_tensor, e_tensor)
    
    # Using persistent_workers can reduce worker re-spawn overhead across epochs
    loader = DataLoader(
        dataset,
        batch_size=1024,
        shuffle=True,
        num_workers=4,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=True  # Changed from False to True
    )
    
    # Set up AMP if supported
    scaler = torch.cuda.amp.GradScaler() if device.type == 'cuda' else None

    for epoch in range(epochs):
        model.train()
        for x, y, e in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            e = e.to(device, non_blocking=True)
            
            optimizer.zero_grad(set_to_none=True)
            # Use automatic mixed precision if available
            if scaler is not None:
                with torch.cuda.amp.autocast():
                    outputs = model(x).squeeze()
                    if model_type == "groupdro":
                        losses = F.binary_cross_entropy(outputs, y, reduction='none')
                        group_losses = [losses[e == g].mean() for g in torch.unique(e)]
                        group_losses = torch.stack([gl for gl in group_losses if not torch.isnan(gl)])
                        weights = torch.ones_like(group_losses) / len(group_losses)
                        loss = torch.sum(weights * group_losses)
                    elif model_type == "vrex":
                        losses = F.binary_cross_entropy(outputs, y, reduction='none')
                        domain_losses = [losses[e == d].mean() for d in torch.unique(e)]
                        if len(domain_losses) >= 2:
                            domain_losses = torch.stack(domain_losses)
                            loss = domain_losses.mean() + 0.1 * domain_losses.var()
                        else:
                            loss = losses.mean()
                    else:
                        loss = F.binary_cross_entropy(outputs, y)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(x).squeeze()
                if model_type == "groupdro":
                    losses = F.binary_cross_entropy(outputs, y, reduction='none')
                    group_losses = [losses[e == g].mean() for g in torch.unique(e)]
                    group_losses = torch.stack([gl for gl in group_losses if not torch.isnan(gl)])
                    weights = torch.ones_like(group_losses) / len(group_losses)
                    loss = torch.sum(weights * group_losses)
                elif model_type == "vrex":
                    losses = F.binary_cross_entropy(outputs, y, reduction='none')
                    domain_losses = [losses[e == d].mean() for d in torch.unique(e)]
                    if len(domain_losses) >= 2:
                        domain_losses = torch.stack(domain_losses)
                        loss = domain_losses.mean() + 0.1 * domain_losses.var()
                    else:
                        loss = losses.mean()
                else:
                    loss = F.binary_cross_entropy(outputs, y)
                
                loss.backward()
                optimizer.step()
    
    # Evaluate on full training data (without gradient computations)
    with torch.no_grad():
        model.eval()
        X_full = X_tensor.to(device)
        reps = model.get_representation(X_full).cpu().numpy()
        hatY = (model(X_full).squeeze().cpu().numpy() > 0.5).astype(int)
    
    return reps, hatY, preprocessor

def extract_representations_and_compute_mi(X_train, X_test, y_train, y_test, domain_train, domain_test, model_type, device, pca_components):
    # Combine train and test sets and preprocess only once
    X_combined = pd.concat([X_train, X_test])
    y_combined = np.concatenate([y_train, y_test])
    domain_combined = np.concatenate([domain_train, domain_test])
    
    preprocessor = create_preprocessor(pca_components)
    X_processed = preprocessor.fit_transform(X_combined)
    
    if model_type == "xgboost":
        model = xgb.XGBClassifier(
            tree_method='gpu_hist' if torch.cuda.is_available() else 'hist',
            predictor='gpu_predictor' if torch.cuda.is_available() else 'cpu_predictor',
            n_jobs=-1
        )
        model.fit(X_processed, y_combined)
        hatY = model.predict(X_processed)
        rep = X_processed
    else:
        reps, hatY, _ = train_pytorch_model(
            X_combined, pd.Series(y_combined),
            pd.Series(domain_combined), model_type,
            device, pca_components
        )
        rep = reps
    
    return compute_mi_terms(rep, hatY, y_combined, domain_combined)

def compute_model_accuracy(model_type, X_train, X_test, y_train, y_test, device, pca_components):
    preprocessor = create_preprocessor(pca_components)
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)
    
    if model_type == "xgboost":
        model = xgb.XGBClassifier(
            tree_method='gpu_hist' if torch.cuda.is_available() else 'hist',
            n_jobs=-1
        )
        model.fit(X_train_processed, y_train)
        return (
            model.score(X_train_processed, y_train),
            model.score(X_test_processed, y_test)
        )
    else:
        model = MLP(pca_components).to(device)
        model = compile_model(model)
        optimizer = optim.Adam(model.parameters())
        
        X_tensor = torch.tensor(X_train_processed, dtype=torch.float32)
        y_tensor = torch.tensor(y_train.values, dtype=torch.float32)
        train_dataset = TensorDataset(X_tensor, y_tensor)
        train_loader = DataLoader(
            train_dataset,
            batch_size=1024,
            shuffle=True,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=True  # Enable persistent workers here as well
        )
        
        # Optionally use AMP for training
        scaler = torch.cuda.amp.GradScaler() if device.type == 'cuda' else None
        
        for _ in range(50):
            model.train()
            for x, y in train_loader:
                x = x.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                
                if scaler is not None:
                    with torch.cuda.amp.autocast():
                        outputs = model(x).squeeze()
                        loss = F.binary_cross_entropy(outputs, y)
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    outputs = model(x).squeeze()
                    loss = F.binary_cross_entropy(outputs, y)
                    loss.backward()
                    optimizer.step()
        
        model.eval()
        with torch.no_grad():
            X_test_tensor = torch.tensor(X_test_processed, dtype=torch.float32).to(device)
            test_preds = (model(X_test_tensor).squeeze().cpu().numpy() > 0.5)
            ood_acc = (test_preds == y_test.values).mean()
            
            train_preds = (model(X_tensor.to(device)).squeeze().cpu().numpy() > 0.5)
            id_acc = (train_preds == y_train.values).mean()
        
        return id_acc, ood_acc

def main(experiment, cache_dir, model_type, pca_components, debug=False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results = {}
    
    # Use an in-memory results list instead of writing to disk on every experiment run
    summary_lines = []
    
    for setting in ["causal", "arguablycausal", "all"]:
        if setting not in experiment:
            continue
            
        dataset = get_dataset(experiment[setting], cache_dir)
        X_train, y_train, _, domain_train = dataset.get_pandas("train")
        X_test, y_test, _, domain_test = dataset.get_pandas("ood_test")
        
        mi_terms = extract_representations_and_compute_mi(
            X_train, X_test, y_train, y_test,
            domain_train, domain_test, model_type,
            device, pca_components
        )
        
        id_acc, ood_acc = compute_model_accuracy(
            model_type, X_train, X_test,
            y_train, y_test, device, pca_components
        )
        
        results[setting] = {
            "Informativeness": float(mi_terms[0]),
            "Invariance": float(mi_terms[1]),
            "Covariate_Shift": float(mi_terms[2]),
            "Concept_Shift": float(mi_terms[3]),
            "Residual": float(mi_terms[4]),
            "ID_Accuracy": float(id_acc),
            "OOD_Accuracy": float(ood_acc),
            "PCA_Components": pca_components
        }
        
        summary = (
            f"Model: {model_type} | Dataset: {experiment['all']} | Setting: {setting}\n"
            f"  Informativeness: {mi_terms[0]:.4f}\n"
            f"  Invariance: {mi_terms[1]:.4f}\n"
            f"  Covariate Shift: {mi_terms[2]:.4f}\n"
            f"  Concept Shift: {mi_terms[3]:.4f}\n"
            f"  Residual: {mi_terms[4]:.4f}\n"
            f"  ID Accuracy: {id_acc:.4f}\n"
            f"  OOD Accuracy: {ood_acc:.4f}\n"
            "----------------------------------------"
        )
        summary_lines.append(summary)
    
    # Write all summaries at once to reduce IO overhead
    with open(OUTPUT_FILE, "a") as f:
        f.write("\n".join(summary_lines) + "\n")
    
    output_file = os.path.join("results", f"{model_type}_{experiment['all']}_results.json")
    with open(output_file, "w") as f:
        json.dump(results, f, indent=4)
    
    # Clean up CUDA memory at the end of processing each experiment
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    return results

if __name__ == "__main__":
    # Use 'spawn' for multiprocessing start method
    torch.multiprocessing.set_start_method('spawn', force=True)
    parser = argparse.ArgumentParser(description="Domain Shift Analysis")
    parser.add_argument("--cache_dir", default="tmp")
    parser.add_argument("--model", required=True, choices=["mlp", "xgboost", "groupdro", "vrex"])
    parser.add_argument("--pca_components", type=int, default=DEFAULT_PCA_COMPONENTS)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    experiments = [
        {"causal": "diabetes_readmission_causal", "arguablycausal": "diabetes_readmission_arguablycausal", "all": "diabetes_readmission"},
        {"causal": "acsfoodstamps_causal", "arguablycausal": "acsfoodstamps_arguablycausal", "all": "acsfoodstamps"},
        {"causal": "acsincome_causal", "arguablycausal": "acsincome_arguablycausal", "anticausal": "acsincome_anticausal", "all": "acsincome"},
        {"causal": "acspubcov_causal", "arguablycausal": "acspubcov_arguablycausal", "all": "acspubcov"},
        {"causal": "acsunemployment_causal", "arguablycausal": "acsunemployment_arguablycausal", "anticausal": "acsunemployment_anticausal", "all": "acsunemployment"},
        {"causal": "brfss_diabetes_causal", "arguablycausal": "brfss_diabetes_arguablycausal", "anticausal": "brfss_diabetes_anticausal", "all": "brfss_diabetes"},
        {"causal": "brfss_blood_pressure_causal", "arguablycausal": "brfss_blood_pressure_arguablycausal", "anticausal": "brfss_blood_pressure_anticausal", "all": "brfss_blood_pressure"},
        {"causal": "college_scorecard_causal", "arguablycausal": "college_scorecard_arguablycausal", "all": "college_scorecard"},
        {"causal": "assistments_causal", "arguablycausal": "assistments_arguablycausal", "all": "assistments"},
    ]

    for exp in experiments:
        try:
            print(f"Processing experiment: {exp['all']}")
            main(exp, args.cache_dir, args.model, args.pca_components, args.debug)
        except Exception as e:
            print(f"Error processing {exp['all']}: {str(e)}")
            print(traceback.format_exc())
