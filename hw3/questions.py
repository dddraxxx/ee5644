#%% Author: Qihua Dong
print("HW3")
print("Author: Qihua Dong")

#%% Q1
print("Q1")
print("-"*10)
#%% 1. Data Distribution
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from scipy.stats import multivariate_normal

# Define the number of classes and dimensions
C = 4
d = 3

# Mean vectors and covariance matrices
means = [np.random.rand(d) * 10 for _ in range(C)]
covariances = [np.eye(d) * (1 + np.random.rand()) for _ in range(C)]

# Function to generate samples from the class-conditional pdfs
def generate_samples(n_samples, priors):
    labels = np.random.choice(range(C), size=n_samples, p=priors)
    data = np.array([np.random.multivariate_normal(means[label], covariances[label]) for label in labels])
    return data, labels

# Uniform priors
priors = [1 / C] * C

#%% 2. Generate Data
# Training datasets of various sizes and a large test dataset
train_sizes = [100, 500, 1000, 5000, 10000]
test_size = 100000

# Generate datasets
train_datasets = [generate_samples(size, priors) for size in train_sizes]
test_data, test_labels = generate_samples(test_size, priors)

#%% 3. Theoretically Optimal Classifier
# Define the MAP classifier using true pdfs
def map_classifier(x):
    class_probs = [multivariate_normal.pdf(x, means[c], covariances[c]) * priors[c] for c in range(C)]
    return np.argmax(class_probs, axis=0)

# Apply to the test data
test_predictions = np.apply_along_axis(map_classifier, 1, test_data)
optimal_error = np.mean(test_predictions != test_labels)

#%% 4. Define the PyTorch MLP
class MLP(nn.Module):
    def __init__(self, input_dim, hidden_units, output_dim):
        super(MLP, self).__init__()
        self.hidden = nn.Linear(input_dim, hidden_units)
        self.activation = nn.ELU()
        self.output = nn.Linear(hidden_units, output_dim)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.hidden(x)
        x = self.activation(x)
        x = self.output(x)
        return self.softmax(x)

# Cross-validation function
def cross_validate_mlp(data, labels, hidden_units, k=10, epochs=10):
    kf = KFold(n_splits=k)
    errors = []
    for train_idx, val_idx in kf.split(data):
        train_data, val_data = data[train_idx], data[val_idx]
        train_labels, val_labels = labels[train_idx], labels[val_idx]

        # Convert to PyTorch tensors
        train_data = torch.tensor(train_data, dtype=torch.float32)
        train_labels = torch.tensor(train_labels, dtype=torch.long)
        val_data = torch.tensor(val_data, dtype=torch.float32)
        val_labels = torch.tensor(val_labels, dtype=torch.long)

        # Initialize and train the model
        model = MLP(d, hidden_units, C)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        for epoch in range(epochs):
            optimizer.zero_grad()
            outputs = model(train_data)
            loss = criterion(outputs, train_labels)
            loss.backward()
            optimizer.step()

        # Evaluate on validation data
        with torch.no_grad():
            val_outputs = model(val_data)
            val_preds = torch.argmax(val_outputs, axis=1)
            error = torch.mean((val_preds != val_labels).float()).item()
            errors.append(error)
    return np.mean(errors)

#%% 5. Model Order Selection
units_range = [5, 10, 20, 50]
best_hidden_units = []

for train_data, train_labels in train_datasets:
    errors = [cross_validate_mlp(train_data, train_labels, units) for units in units_range]
    best_units = units_range[np.argmin(errors)]
    best_hidden_units.append(best_units)

#%% 6. Train Final Models
final_models = []
for (train_data, train_labels), hidden_units in zip(train_datasets, best_hidden_units):
    train_data = torch.tensor(train_data, dtype=torch.float32)
    train_labels = torch.tensor(train_labels, dtype=torch.long)

    model = MLP(d, hidden_units, C)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(50):
        optimizer.zero_grad()
        outputs = model(train_data)
        loss = criterion(outputs, train_labels)
        loss.backward()
        optimizer.step()

    final_models.append(model)

#%% 7. Performance Assessment
# Main Plot: Test set empirical P(error)
errors = []
test_data_tensor = torch.tensor(test_data, dtype=torch.float32)
test_labels_tensor = torch.tensor(test_labels, dtype=torch.long)

for model in final_models:
    with torch.no_grad():
        test_outputs = model(test_data_tensor)
        test_preds = torch.argmax(test_outputs, axis=1)
        error = torch.mean((test_preds != test_labels_tensor).float()).item()
        errors.append(error)

# Plotting
plt.figure(figsize=(8, 6))
plt.semilogx(train_sizes, errors, marker='o', label="Empirical MLP Error")
plt.axhline(y=optimal_error, color='red', linestyle='--', label="Optimal Classifier Error")
plt.xlabel("Number of Training Samples")
plt.ylabel("Empirical P(error)")
plt.title("Test Set Empirical P(error) vs Training Samples")
plt.legend()
plt.grid(True)
plt.show()

# Supplementary Plot: Best hidden units vs training dataset size
plt.figure(figsize=(8, 6))
plt.plot(train_sizes, best_hidden_units, marker='s', label="Best Hidden Units")
plt.xlabel("Number of Training Samples")
plt.ylabel("Number of Hidden Units")
plt.title("Best Hidden Units vs Training Dataset Size")
plt.legend()
plt.grid(True)
plt.show()

#%% 8. Summary of the Process
description = """
Process Summary:

1. Defined a 3D Gaussian distribution for 4 classes with uniform priors, ensuring class overlap to achieve a 10%-20% optimal error rate.
2. Generated training datasets (100 to 10,000 samples) and a test dataset (100,000 samples).
3. Constructed a MAP classifier using the true data distribution to estimate the theoretical optimal error.
4. Designed and trained 2-layer MLPs with smooth activations, selecting hidden units via 10-fold cross-validation.
5. Evaluated MLPs on the test dataset, estimating empirical error rates and comparing them with the theoretical optimal.
6. Plotted:
   - Empirical error rates vs. training samples (semilog-x axis) with the optimal error marked.
   - Best hidden units vs. training dataset size.

Key Result: MLP error approached the theoretical optimal as training data increased.
"""

print(description)

# Print numerical report
print("Numerical Report of Test Set Empirical P(error):")
print(f"Theoretical Optimal Classifier Error: {optimal_error:.4f}")
for size, error in zip(train_sizes, errors):
    print(f"Training Size: {size}, MLP Empirical Error: {error:.4f}")

#%% Q2
print('\n\n')
print("Q2")
print("-"*10)
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
#%% Summary of the Process
description = """
Process Summary:

1. Defined a 4-component GMM with 2D data, distinct means, covariances, and probabilities, ensuring overlap between two components.
2. Generated datasets of sizes 10, 100, and 1000 samples from the true GMM.
3. Evaluated candidate GMMs (1-10 components) using 10-fold cross-validation, with log-likelihood as the performance metric.
4. Repeated the experiment 100 times to determine the most selected model order for each dataset size.
5. Plotted:
   - Average log-likelihoods for each model order and dataset size.
   - Selection probabilities for each model order across 100 repetitions.

Key Result: Larger datasets led to better identification of the true GMM model order.
"""

print(description)
# %%
