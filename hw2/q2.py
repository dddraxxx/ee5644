#%% Common Code: Import Libraries and Define Parameters
import numpy as np
from matplotlib import pyplot as plt
import pylab
from mpl_toolkits.mplot3d import Axes3D

# Parameters
N_train_100 = 100
N_train_1000 = 1000
gamma_values = np.logspace(-3, 3, 7)  # Range of gamma values for MAP estimation

#%% Function Definitions: Data Generation and Plotting Functions

def generateData(N):
    gmmParameters = {
        'priors': [.3, .4, .3],
        'meanVectors': np.array([[-10, 0, 10], [0, 0, 0], [10, 0, -10]]),
        'covMatrices': np.zeros((3, 3, 3))
    }
    gmmParameters['covMatrices'][:, :, 0] = np.array([[1, 0, -3], [0, 1, 0], [-3, 0, 15]])
    gmmParameters['covMatrices'][:, :, 1] = np.array([[8, 0, 0], [0, .5, 0], [0, 0, .5]])
    gmmParameters['covMatrices'][:, :, 2] = np.array([[1, 0, -3], [0, 1, 0], [-3, 0, 15]])

    x, labels = generateDataFromGMM(N, gmmParameters)
    return x

def generateDataFromGMM(N, gmmParameters):
    priors = gmmParameters['priors']
    meanVectors = gmmParameters['meanVectors']
    covMatrices = gmmParameters['covMatrices']
    n = meanVectors.shape[0]  # Data dimensionality
    C = len(priors)  # Number of components
    x = np.zeros((n, N))
    labels = np.zeros((1, N))
    u = np.random.random((1, N))
    thresholds = np.zeros((1, C + 1))
    thresholds[:, 0:C] = np.cumsum(priors)
    thresholds[:, C] = 1
    for l in range(C):
        indl = np.where(u <= float(thresholds[:, l]))
        Nl = len(indl[1])
        labels[indl] = (l + 1) * 1
        u[indl] = 1.1
        x[:, indl[1]] = np.transpose(np.random.multivariate_normal(meanVectors[:, l], covMatrices[:, :, l], Nl))

    return x, labels

# def plot3(a, b, c, mark="o", col="b", title="3D Data Plot"):
#     fig = pylab.figure()
#     ax = Axes3D(fig)
#     ax.scatter(a, b, c, marker=mark, color=col)
#     ax.set_xlabel("x1")
#     ax.set_ylabel("x2")
#     ax.set_zlabel("y")
    ax.set_title(title)
def plot3(a, b, c, mark="o", col="b", title="3D Data Plot"):
    plt.ion()  # Enable interactive mode
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(a, b, c, marker=mark, color=col)
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.set_zlabel("y")
    ax.set_title(title)
    plt.show()

#%% Q2: Part 2.1 - Data Generation for Training and Validation Sets
print("\n--- Q2: Part 2.1 - Generating Training and Validation Data ---")

# Generate training data
data_train = generateData(N_train_100)
xTrain = data_train[:2, :]  # xTrain contains first two rows (x1, x2)
yTrain = data_train[2, :]    # yTrain contains third row (y values)
plot3(xTrain[0, :], xTrain[1, :], yTrain, title="Training Dataset (100 Samples)")

# Generate validation data
data_validate = generateData(N_train_1000)
xValidate = data_validate[:2, :]  # xValidate contains first two rows (x1, x2)
yValidate = data_validate[2, :]    # yValidate contains third row (y values)
plot3(xValidate[0, :], xValidate[1, :], yValidate, title="Validation Dataset (1000 Samples)")

#%% Q2: Part 2.2 - Maximum Likelihood (ML) Estimation for w
print("\n--- Q2: Part 2.2 - Performing Maximum Likelihood (ML) Estimation ---")

def cubic_model(x, w):
    """ Define cubic polynomial model c(x, w) = w0 + w1*x1 + w2*x2 + ... """
    x1, x2 = x
    return w[0] + w[1]*x1 + w[2]*x2 + w[3]*x1**2 + w[4]*x1*x2 + w[5]*x2**2 + \
           w[6]*x1**3 + w[7]*x1**2*x2 + w[8]*x1*x2**2 + w[9]*x2**3

# Initialize weights randomly
w_ml = np.random.randn(10)

# Define loss function for ML
def ml_loss(w, x, y):
    predictions = cubic_model(x, w)
    return np.mean((y - predictions) ** 2)

# Optimize using gradient descent
from scipy.optimize import minimize
result_ml = minimize(ml_loss, w_ml, args=(xTrain, yTrain), method='BFGS')
w_ml_estimated = result_ml.x
print("Estimated weights (w) from ML:", w_ml_estimated)

#%% Q2: Part 2.3 - Maximum a Posteriori (MAP) Estimation for w
print("\n--- Q2: Part 2.3 - Performing Maximum a Posteriori (MAP) Estimation ---")

def map_loss(w, x, y, gamma):
    return ml_loss(w, x, y) + (gamma * np.sum(w ** 2))  # Add regularization term for MAP

w_map_results = {}
for gamma in gamma_values:
    result_map = minimize(map_loss, w_ml, args=(xTrain, yTrain, gamma), method='BFGS')
    w_map_results[gamma] = result_map.x
    print(f"Estimated weights (w) from MAP with gamma={gamma}:", result_map.x)

#%% Q2: Part 4.1 - Validation Error Calculation for ML and MAP
print("\n---Q2: Part 4.1 - Calculating Validation Error ---")

# Calculate ML error on validation set
y_pred_ml = cubic_model(xValidate, w_ml_estimated)
ml_error = np.mean((yValidate - y_pred_ml) ** 2)
print("Validation Error for ML:", ml_error)

# Calculate MAP errors for each gamma value on validation set
map_errors = {}
for gamma, w_map in w_map_results.items():
    y_pred_map = cubic_model(xValidate, w_map)
    map_errors[gamma] = np.mean((yValidate - y_pred_map) ** 2)
    print(f"Validation Error for MAP with gamma={gamma}:", map_errors[gamma])

#%% Q2: Part 4.2 - Description and Analysis
print("\n--- Q2: Part 4.2 - Description and Analysis ---")
print("The validation error for the ML estimator provides a baseline for comparing performance.")
print("For the MAP estimator, different values of gamma show varying performance, reflecting the balance between data fitting and regularization.")
print("As gamma increases, the MAP estimator tends to regularize the weights more heavily, which can reduce overfitting but may also result in underfitting if gamma is too large.")
print("This analysis helps in understanding the trade-off in choosing gamma for the MAP estimator based on model performance on validation data.")

# %%
