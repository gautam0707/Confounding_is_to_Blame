import numpy as np
from scipy import stats
import pandas as pd

# Define your data structure
data = {
    "Readmission":       [(0.107, 0.002), (0.068, 0.002), (0.097, 0.002), (2.032, 0.000)],
    "Food Stamps":       [(0.126, 0.004), (0.030, 0.003), (0.108, 0.001), (2.118, 0.001)],
    "Income":            [(0.168, 0.002), (0.075, 0.003), (0.147, 0.001), (2.059, 0.002)],
    "Public Coverage":   [(0.231, 0.002), (0.412, 0.006), (0.222, 0.002), (1.945, 0.001)],
    "Unemployment":      [(0.117, 0.001), (0.019, 0.002), (0.114, 0.002), (2.010, 0.003)],
    "Diabetes":          [(0.032, 0.002), (0.048, 0.001), (0.022, 0.002), (2.132, 0.001)],
    "Hypertension":      [(0.090, 0.002), (0.183, 0.003), (0.037, 0.001), (1.883, 0.004)],
    "ASSISTments":       [(0.293, 0.002), (0.260, 0.001), (0.306, 0.002), (0.367, 0.004)],
}

shift_types = [
    "Conditional Covariate Shift",
    "Label Shift",
    "Covariate Shift",
    "Concept Shift"
]

# Initialize results storage
results = {shift: [] for shift in shift_types}

# Generate samples and populate results
n_seeds = 10  # As per the table description
for dataset in data:
    for idx, (mean, std) in enumerate(data[dataset]):
        # Generate 5 samples with the given mean and std
        samples = np.random.normal(loc=mean, scale=std, size=n_seeds)
        results[shift_types[idx]].extend(samples)

# Perform t-tests and print results
alpha = 0.05
for shift in shift_types:
    t_stat, p_val = stats.ttest_1samp(results[shift], popmean=0)
    
    print(f"=== {shift} ===")
    print(f"T-statistic: {t_stat:.4f}")
    print(f"P-value: {p_val:.4f}")
    if p_val < alpha:
        print(f"Conclusion: Mean is significantly different from zero (p < {alpha})")
    else:
        print(f"Conclusion: No significant evidence against mean = 0 (p >= {alpha})")
    print("\n")

# Optional: Save results to CSV

# pd.DataFrame(results).to_csv("shift_type_samples.csv", index=False)