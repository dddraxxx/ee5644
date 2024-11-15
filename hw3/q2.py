#%% 1. Define the True GMM
# Define a GMM with 4 components, distinct means, covariances, and probabilities.
import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
import pandas as pd

# True GMM parameters
n_components_true = 4  # Number of Gaussian components
d = 2  # Dimensionality of data

# Define means, covariances, and component probabilities
means = [np.array([i * 2, i * 3]) for i in range(n_components_true)]
covariances = [np.array([[1, 0.5], [0.5, 1]]) * (1 + i / 2) for i in range(n_components_true)]
component_probs = [0.4, 0.3, 0.2, 0.1]

# Function to generate samples from the GMM
def generate_gmm_samples(n_samples, means, covariances, probs):
    labels = np.random.choice(range(len(means)), size=n_samples, p=probs)
    data = np.array([np.random.multivariate_normal(means[label], covariances[label]) for label in labels])
    return data, labels

#%% 2. Generate Multiple Datasets
# Generate datasets with sizes 10, 100, and 1000 samples.
dataset_sizes = [10, 100, 1000]
datasets = [generate_gmm_samples(size, means, covariances, component_probs) for size in dataset_sizes]

#%% 3. Evaluate GMMs with Cross-Validation
# Evaluate candidate GMMs (1–10 components) using 10-fold cross-validation and log-likelihood.

def evaluate_gmm(data, max_components=10, n_splits=10):
    kf = KFold(n_splits=n_splits)
    n_samples = len(data)
    log_likelihoods = np.zeros((max_components, n_splits))
    log_likelihoods[:] = np.nan  # Initialize with NaNs to handle invalid folds

    for n_components in range(1, max_components + 1):
        for fold_idx, (train_idx, val_idx) in enumerate(kf.split(data)):
            # Skip if training fold size is less than the number of components
            if len(train_idx) < n_components:
                continue

            train_data, val_data = data[train_idx], data[val_idx]
            gmm = GaussianMixture(n_components=n_components, covariance_type='full', max_iter=200, random_state=0)

            try:
                gmm.fit(train_data)
                log_likelihoods[n_components - 1, fold_idx] = gmm.score(val_data)
            except ValueError as e:
                # Handle cases where GMM fails internally
                log_likelihoods[n_components - 1, fold_idx] = np.nan

    # Replace rows with NaNs if no valid folds exist for a specific component number
    valid_rows = ~np.isnan(log_likelihoods).all(axis=1)
    avg_log_likelihoods = np.full(max_components, np.nan)  # Initialize with NaN
    avg_log_likelihoods[valid_rows] = np.nanmean(log_likelihoods[valid_rows], axis=1)

    return avg_log_likelihoods

# Evaluate GMMs for each dataset size
max_components = 10
evaluation_results = [evaluate_gmm(data, max_components=max_components) for data, _ in datasets]

#%% 4. Repeat Experiment and Summarize
# Repeat the evaluation 100 times to determine the most selected model order.
n_repeats = 100
selection_counts = np.zeros((len(dataset_sizes), max_components))

from tqdm import tqdm
for repeat in tqdm(range(n_repeats)):
    for dataset_idx, (data, _) in enumerate(datasets):
        avg_log_likelihoods = evaluate_gmm(data, max_components=max_components)
        best_model_order = np.argmax(avg_log_likelihoods) + 1  # Add 1 because indices are 0-based
        selection_counts[dataset_idx, best_model_order - 1] += 1

# Normalize selection counts to get probabilities
selection_probs = selection_counts / n_repeats

#%% Develop Tables and Figures to Summarize Results
# Create tables and figures for the selection probabilities.

# Inform the reader about the upcoming table plots
print("The following tables report the selection probabilities for each dataset size,")
print("indicating how often each model order was selected as the best model over multiple repetitions.")

# Tables of selection probabilities
for idx, size in enumerate(dataset_sizes):
    df = pd.DataFrame({
        'Number of Components': range(1, max_components + 1),
        'Selection Probability': selection_probs[idx]
    })
    print(f"\nSelection Probabilities for Dataset Size {size}:\n")
    print(df.to_string(index=False))

# Figures: Plot average log-likelihoods and selection probabilities
# Inform the reader about the upcoming average log-likelihood plots
print("\nThe following plots show the average log-likelihoods for different model orders,")
print("helping to identify the performance of each model order for each dataset size.")

# Plot average log-likelihoods for different model orders
for idx, size in enumerate(dataset_sizes):
    plt.figure(figsize=(8, 6))
    plt.plot(range(1, max_components + 1), evaluation_results[idx], marker='o')
    plt.xlabel("Number of Components")
    plt.ylabel("Average Log-Likelihood")
    plt.title(f"GMM Model Order Selection (Dataset Size: {size})")
    plt.grid(True)
    plt.show()

# Inform the reader about the upcoming selection probability plots
print("\nThe following plots display the selection probabilities for each model order,")
print("illustrating how often each model order was chosen as the best model for each dataset size.")

# Plot selection probabilities
for idx, size in enumerate(dataset_sizes):
    plt.figure(figsize=(8, 6))
    plt.bar(range(1, max_components + 1), selection_probs[idx])
    plt.xlabel("Number of Components")
    plt.ylabel("Selection Probability")
    plt.title(f"Model Order Selection Probabilities (Dataset Size: {size})")
    plt.grid(True)
    plt.show()

# %%
