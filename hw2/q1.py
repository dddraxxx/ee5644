#%% Common Code: Define Parameters and Generate Datasets
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
from sklearn.metrics import roc_curve
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

# Define class priors
P_L0 = 0.6
P_L1 = 0.4

# Define weights
w_01 = w_02 = w_11 = w_12 = 0.5  # weights

# Mean vectors and covariance matrices for each class component
m_01 = np.array([-0.9, -1.1])
m_02 = np.array([0.8, 0.75])
m_11 = np.array([-1.1, 0.9])
m_12 = np.array([0.9, -0.75])
C_01 = C_02 = C_11 = C_12 = np.array([[0.75, 0], [0, 1.25]])

# Dataset sizes
D20_train_size = 20
D200_train_size = 200
D2000_train_size = 2000
D10K_validate_size = 10000

# Function to generate data based on class priors and conditional pdfs
def generate_data(size, P_L0, m_01, m_02, m_11, m_12, C_01, C_02, C_11, C_12):
    data, labels = [], []
    for _ in range(size):
        # Select class label based on priors
        L = np.random.choice([0, 1], p=[P_L0, P_L1])
        if L == 0:
            sample = (multivariate_normal.rvs(mean=m_01, cov=C_01) if np.random.rand() < 0.5
                      else multivariate_normal.rvs(mean=m_02, cov=C_02))
        else:
            sample = (multivariate_normal.rvs(mean=m_11, cov=C_11) if np.random.rand() < 0.5
                      else multivariate_normal.rvs(mean=m_12, cov=C_12))
        data.append(sample)
        labels.append(L)
    return np.array(data), np.array(labels)

# Generate datasets for training and validation
D20_train, D20_labels = generate_data(D20_train_size, P_L0, m_01, m_02, m_11, m_12, C_01, C_02, C_11, C_12)
D200_train, D200_labels = generate_data(D200_train_size, P_L0, m_01, m_02, m_11, m_12, C_01, C_02, C_11, C_12)
D2000_train, D2000_labels = generate_data(D2000_train_size, P_L0, m_01, m_02, m_11, m_12, C_01, C_02, C_11, C_12)
D10K_validate, D10K_labels = generate_data(D10K_validate_size, P_L0, m_01, m_02, m_11, m_12, C_01, C_02, C_11, C_12)

#%% Part 1: Theoretically Optimal Classifier and ROC Curve (6%)

## 1.1 Define the Optimal Classifier
def optimal_classifier(x):
    """
    Optimal Bayesian classifier based on provided priors and pdfs.
    """
    # Calculate the conditional densities for each class
    p_x_given_L0 = (w_01 * multivariate_normal.pdf(x, mean=m_01, cov=C_01) +
                    w_02 * multivariate_normal.pdf(x, mean=m_02, cov=C_02))
    p_x_given_L1 = (w_11 * multivariate_normal.pdf(x, mean=m_11, cov=C_11) +
                    w_12 * multivariate_normal.pdf(x, mean=m_12, cov=C_12))

    # Calculate posterior probabilities
    p_L0_given_x = P_L0 * p_x_given_L0
    p_L1_given_x = P_L1 * p_x_given_L1

    # Classify based on maximum posterior
    return 0 if p_L0_given_x > p_L1_given_x else 1

## 1.2 Apply the Classifier to Validation Dataset
print("Q1: Part 1.2 - Applying the Optimal Classifier to Validation Dataset")
optimal_predictions = np.array([optimal_classifier(x) for x in D10K_validate])

## 1.3 Estimate and Plot ROC Curve
print("Q1: Part 1.3 - ROC Curve and Minimum P(Error) Point")
fpr, tpr, thresholds = roc_curve(D10K_labels, optimal_predictions)
plt.figure()
plt.plot(fpr, tpr, label="ROC Curve")
plt.scatter([fpr[0]], [tpr[0]], marker='o', color="red", label="Min P(Error) Point")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend()
plt.title("ROC Curve with Min P(Error) Point")
plt.show()

## 1.4 Report Minimum Probability of Error
min_p_error = np.mean(optimal_predictions != D10K_labels)
print("Q1: Part 1.4 - Minimum Probability of Error (P(error)):", min_p_error)

#%% Part 2: Approximate Classifiers Based on Logistic Models (12%)

# Define training datasets in dictionary form
train_datasets = {
    'D20': (D20_train, D20_labels),
    'D200': (D200_train, D200_labels),
    'D2000': (D2000_train, D2000_labels)
}

### 2.1 Train Logistic-Linear Classifiers
print("Q1: Part 2.1 - Training Logistic-Linear Classifiers")
linear_models = {}

for dataset_name, (X_train, y_train) in train_datasets.items():
    # Create and fit the logistic-linear model
    model = LogisticRegression(solver='lbfgs')
    model.fit(X_train, y_train)
    linear_models[dataset_name] = model

    # 2.2 Evaluate Logistic-Linear Classifiers on Validation Set
    y_pred = model.predict(D10K_validate)
    p_error = np.mean(y_pred != D10K_labels)
    print(f"Q1: Part 2.2 - {dataset_name} Logistic-Linear P(error):", p_error)

### 2.3 Train Logistic-Quadratic Classifiers
print("Q1: Part 2.3 - Training Logistic-Quadratic Classifiers")
quadratic_models = {}

for dataset_name, (X_train, y_train) in train_datasets.items():
    # Create and fit the logistic-quadratic model
    model = make_pipeline(PolynomialFeatures(degree=2), LogisticRegression(solver='lbfgs'))
    model.fit(X_train, y_train)
    quadratic_models[dataset_name] = model

    # 2.4 Evaluate Logistic-Quadratic Classifiers on Validation Set
    y_pred = model.predict(D10K_validate)
    p_error = np.mean(y_pred != D10K_labels)
    print(f"Q1: Part 2.4 - {dataset_name} Logistic-Quadratic P(error):", p_error)

#%% Discussion: Performance Analysis and Comparison

print("\n--- Q1: Discussion ---")
print("The performance of classifiers trained on increasing dataset sizes improves as expected, reflecting "
      "the general trend that more data supports more accurate model estimation.")
print("Between the logistic-linear and logistic-quadratic classifiers, the quadratic models offer better "
      "approximation as they allow for more complex decision boundaries, closer to the optimal Bayesian classifier.")
print("Compared to the optimal classifier, all approximations yield higher error rates. However, as sample sizes "
      "increase, logistic models show reduced errors, approaching the performance of the optimal classifier.")

# %%
