#%% Common Code: Import Libraries and Define Parameters
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from mpl_toolkits.mplot3d import Axes3D

# Set random seed for reproducibility
np.random.seed(0)

# Parameters
sigma_x = 0.25
sigma_y = 0.25
measurement_noise_std = 0.3  # Standard deviation of measurement noise
true_position = np.array([0.5, -0.5])  # True vehicle position within unit circle

#%% Function Definitions: Distance and MAP Objective Functions

def generate_landmarks(K):
    """ Generate K landmarks evenly spaced around a unit circle centered at the origin. """
    angles = np.linspace(0, 2 * np.pi, K, endpoint=False)
    return np.stack([np.cos(angles), np.sin(angles)], axis=1)

def generate_measurements(true_position, landmarks, noise_std):
    """ Generate noisy range measurements from the true position to each landmark. """
    distances = np.linalg.norm(landmarks - true_position, axis=1)
    noisy_distances = distances + np.random.normal(0, noise_std, size=distances.shape)
    # Ensure all measurements are non-negative
    noisy_distances = np.maximum(noisy_distances, 0)
    return noisy_distances

def map_objective(position, landmarks, measurements, noise_std, sigma_x, sigma_y):
    """ Calculate the MAP objective function for the vehicle's position. """
    x, y = position
    prior_term = (x ** 2 / (2 * sigma_x ** 2)) + (y ** 2 / (2 * sigma_y ** 2))
    likelihood_term = np.sum(((measurements - np.linalg.norm(landmarks - position, axis=1)) ** 2) / (2 * noise_std ** 2))
    return prior_term + likelihood_term

#%% Q3: Part 2.1 - Implement MAP Estimation for Different Landmark Counts
print("\n--- Q3: Part 2.1 - Implementing MAP Estimation for Different Landmark Counts ---")

landmark_counts = [1, 2, 3, 4]
estimated_positions = {}

for K in landmark_counts:
    print(f"\n--- MAP Estimation with K = {K} landmarks ---")
    landmarks = generate_landmarks(K)
    measurements = generate_measurements(true_position, landmarks, measurement_noise_std)

    # Initial guess for the position (center of the unit circle)
    initial_position = np.array([0.0, 0.0])

    # Optimize MAP objective function
    result = minimize(map_objective, initial_position, args=(landmarks, measurements, measurement_noise_std, sigma_x, sigma_y))
    estimated_positions[K] = result.x
    print(f"Estimated Position (x, y) for K = {K}:", result.x)

#%% Q3: Part 3.1 - Contour Plots of MAP Objective Function for Each K
print("\n--- Q3: Part 3.1 - Contour Plots of MAP Objective Function ---")

x_vals = np.linspace(-2, 2, 100)
y_vals = np.linspace(-2, 2, 100)
X, Y = np.meshgrid(x_vals, y_vals)

for K in landmark_counts:
    landmarks = generate_landmarks(K)
    measurements = generate_measurements(true_position, landmarks, measurement_noise_std)

    # Calculate MAP objective values over the grid
    Z = np.array([
        map_objective([x, y], landmarks, measurements, measurement_noise_std, sigma_x, sigma_y)
        for x, y in zip(X.ravel(), Y.ravel())
    ]).reshape(X.shape)

    # Plot contour map
    plt.figure()
    contour = plt.contourf(X, Y, Z, levels=50, cmap="viridis")
    plt.colorbar(contour)
    plt.scatter(*true_position, color="red", label="True Position", marker="x")
    plt.scatter(landmarks[:, 0], landmarks[:, 1], color="blue", label="Landmarks")
    plt.scatter(*estimated_positions[K], color="yellow", label="Estimated Position", marker="o")
    plt.title(f"MAP Objective Contour Plot for K = {K}")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.show()

#%% Q3: Part 4.2 - Discussion and Analysis
print("\n--- Q3: Part 4.2 - Discussion and Analysis ---")
print("With increasing landmark count (K), the MAP estimate becomes more accurate and certain, "
      "as reflected in the contour plots.")
print("For K=1, the contours are wider, indicating less certainty in the estimate.")
print("As K increases, the contours narrow and concentrate around the true position, suggesting increased confidence "
      "in the estimated position with more landmarks. This is expected as additional measurements provide "
      "more information, allowing for a more precise estimate of the vehicle's location.")
