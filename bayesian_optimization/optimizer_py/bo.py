# Import C++ bindings
import sys
sys.path.append('../Release')
import simulator

# Import libraries for optimization
import torch
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_mll
from botorch.acquisition import LogExpectedImprovement
from botorch.optim import optimize_acqf
from gpytorch.mlls import ExactMarginalLogLikelihood

# Import standard libraries
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import os

# Set device for PyTorch (use GPU if available)
device = torch.device("cpu")
print(f"Using device: {device}")

# Field parameters (fixed for this optimization)
K = 3 # Number of coils
I = 1E5 # Current in Amperes

# FIELD HYPERPARAMETERS SETUP
R_BOUNDS = (0.1, 2.0) # Coil radius bounds in meters
X_BOUNDS = (1.0, 4.0) # Bound for distance of coil centers from origin in meters
THETA_BOUNDS = (0, np.pi) # Bounds for azimuthal angles in spherical coordinates
PHI_BOUNDS = (0, 2*np.pi) # Bounds for polar angles in spherical coordinates

# Define bounds tensor
LOWER = torch.tensor(
    [R_BOUNDS[0]] + [X_BOUNDS[0]]*K + [THETA_BOUNDS[0]]*K + [PHI_BOUNDS[0]]*K + [THETA_BOUNDS[0]]*K + [PHI_BOUNDS[0]]*K,
    dtype=torch.double
).to(device)
UPPER = torch.tensor(
    [R_BOUNDS[1]] + [X_BOUNDS[1]]*K + [THETA_BOUNDS[1]]*K + [PHI_BOUNDS[1]]*K + [THETA_BOUNDS[1]]*K + [PHI_BOUNDS[1]]*K,
    dtype=torch.double
).to(device)

# Define unitary bounds
D = 1 + 5 * K
unit_bounds = torch.zeros(2, D, dtype=torch.double, device=device)
unit_bounds[1] = 1.0  # upper bounds all 1

# Normalization functions to scale parameters to [0, 1] for optimization
def normalize(X):
    """Scale X from original bounds to [0, 1]."""
    # Move to numpy to avoid torch issues
    UPPER_np = UPPER.cpu().numpy()
    LOWER_np = LOWER.cpu().numpy()
    return (X - LOWER_np) / (UPPER_np - LOWER_np)

def denormalize(X):
    """Scale X from [0, 1] back to original bounds."""
    # Move to numpy to avoid torch issues
    UPPER_np = UPPER.cpu().numpy()
    LOWER_np = LOWER.cpu().numpy()
    return X * (UPPER_np - LOWER_np) + LOWER_np

# SIMULATION AND OPTIMIZATION HYPERPARAMETERS
INIT = 3 # Number of initial random samples for BO
MAX_ITER = 20
PATIENCE = 5 # Patience for early stopping
SEED = 42
rng = np.random.default_rng(SEED)

def spherical_to_cartesian(r, theta, phi):
    """
    Convert spherical coordinates to cartesian.
    Convention: theta = azimuth (from X-axis in XY-plane), phi = polar (from Z-axis)
    
    Args:
        r:     radial distance, shape (K,)
        theta: azimuth angle, shape (K,)
        phi:   polar angle,   shape (K,)
    Returns:
        xyz:   cartesian coordinates, shape (K, 3)
    """
    x = r * np.sin(phi) * np.cos(theta)
    y = r * np.sin(phi) * np.sin(theta)
    z = r * np.cos(phi)
    return np.stack([x, y, z], axis=1)

def objective_function(seed, R, centers, normals):
    """
    Calls the C++ simulator to compute the energy hitting the detector.
    R: is coils radius
    centers: list of coil centers (spherical coordinates: r, theta, phi)
    normals: list of coil normal vectors (spherical coordinates: theta, phi)
    Returns: energy hitting the detector (lower is better)
    """
    # Change coordinates from spherical to cartesian for simulator input
    centers_cart = spherical_to_cartesian(centers[:, 0], centers[:, 1], centers[:, 2]) # shape: (K, 3)
    normals_cart = spherical_to_cartesian(np.ones(K), normals[:, 0], normals[:, 1]) # shape: (K, 3) - unit vectors for normals
    # Call the C++ simulator
    expected_energy = simulator.launch_simulation(seed, K, I, R, centers_cart, normals_cart)
    # Take the log of energy to stabilize optimization and handle wide range of values
    return np.log(expected_energy + 1e-8) # Add small value to avoid log(0)

def random_coil_configuration(K):
    """
    Generates random coil configuration in spherical coordinates.
    """
    R = rng.uniform(R_BOUNDS[0], R_BOUNDS[1]) # Random coil radius
    # --- Centers (sampled in spherical, returned in cartesian) ---
    center_r     = rng.uniform(X_BOUNDS[0],     X_BOUNDS[1],     size=K)
    center_theta = rng.uniform(THETA_BOUNDS[0],  THETA_BOUNDS[1], size=K)
    center_phi   = rng.uniform(PHI_BOUNDS[0],    PHI_BOUNDS[1],   size=K)
    centers = np.column_stack((center_r, center_theta, center_phi))
    # --- Normals (unit vectors — r=1, sampled in spherical, returned in cartesian) ---
    normal_theta = rng.uniform(THETA_BOUNDS[0], THETA_BOUNDS[1], size=K)
    normal_phi   = rng.uniform(PHI_BOUNDS[0],   PHI_BOUNDS[1],   size=K)
    normals = np.column_stack((normal_theta, normal_phi))
    return R, centers, normals

def initialize_bo(n_initial_points=INIT):
    """
    Initializes the Bayesian Optimization process with random samples.
    Returns initial data (R, centers, normals, energy).
    """
    R_samples = []
    centers_samples = []
    normals_samples = []
    energy_samples = []
    
    for _ in range(n_initial_points):
        R, centers, normals = random_coil_configuration(K)
        energy = objective_function(rng.integers(1e6), R, centers, normals)
        
        R_samples.append(R)
        centers_samples.append(centers)
        normals_samples.append(normals)
        energy_samples.append(energy)
    
    return np.array(R_samples), np.array(centers_samples), np.array(normals_samples), np.array(energy_samples)

def main():
    # Step 1: Initialize BO with random samples
    R_init, centers_init, normals_init, energy_init = initialize_bo()

    ## Step 2: Fit Gaussian Process model over R, centers and normals
    X_train = R_init.reshape(-1, 1)
    X_train = np.hstack((X_train, centers_init.reshape(-1, 3*K)))
    X_train = np.hstack((X_train, normals_init.reshape(-1, 2*K)))
    #X_train = normalize(X_train) # Normalize training data to [0, 1]
    y_train = -energy_init.flatten() # Negate energy for maximization

    # Fit GP model
    gp = SingleTaskGP(
        torch.tensor(X_train, dtype=torch.double).to(device), 
        torch.tensor(y_train, dtype=torch.double).unsqueeze(-1).to(device)) # Unsqueeze for correct shape
    mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
    fit_gpytorch_mll(mll)

    # Step 3: BO loop
    best_energy = energy_init.min()
    patience_counter = 0
    for iteration in range(MAX_ITER):
        # 1. Optimize acquisition function to find next point
        ei = LogExpectedImprovement(gp, best_f=torch.tensor(y_train.max(), dtype=torch.double).to(device))
        candidate, _ = optimize_acqf(
            acq_function=ei,
            #bounds=unit_bounds,
            bounds=torch.tensor([
                [R_BOUNDS[0]] + [X_BOUNDS[0]]*K + [THETA_BOUNDS[0]]*K + [PHI_BOUNDS[0]]*K + [THETA_BOUNDS[0]]*K + [PHI_BOUNDS[0]]*K,
                [R_BOUNDS[1]] + [X_BOUNDS[1]]*K + [THETA_BOUNDS[1]]*K + [PHI_BOUNDS[1]]*K + [THETA_BOUNDS[1]]*K + [PHI_BOUNDS[1]]*K,
                ], dtype=torch.double, device=device),
            q=1,
            num_restarts=5,
            raw_samples=20,
        )
        
        
        # 2. Extract params from candidate and generate random centers and normals
        #candidate = denormalize(candidate.cpu().numpy()) # Denormalize candidate to original scale
        # Extract R
        R_next = candidate[0, 0].item() # Coil radius
        # Extract centers in spherical coordinates
        center_r     = candidate[0, 1      : 1+K  ].detach().numpy()   # radial distance (norm)
        center_theta = candidate[0, 1+K    : 1+2*K].detach().numpy()   # azimuth
        center_phi   = candidate[0, 1+2*K  : 1+3*K].detach().numpy()   # polar
        centers_next = np.stack([center_r, center_theta, center_phi], axis=1) # shape: (K, 3)
        # Extract normals in spherical coordinates
        normal_theta = candidate[0, 1+3*K  : 1+4*K].detach().numpy()   # azimuth
        normal_phi   = candidate[0, 1+4*K  : 1+5*K].detach().numpy()   # polar
        normals_next = np.stack([normal_theta, normal_phi], axis=1) # shape: (K, 2)
        
        # 3. Evaluate simulation at new point
        energy_next = objective_function(rng.integers(1e6), R_next, centers_next, normals_next)
        
        # 4. Update training data
        X_train = np.vstack((X_train, np.hstack((R_next, centers_next.flatten(), normals_next.flatten()))))
        #X_train = normalize(X_train) # Normalize training data to [0, 1]
        y_train = np.append(y_train, float(-energy_next)) # Negate energy for maximization
        # Check shapes before fitting GP
        assert X_train.shape[0] == y_train.shape[0], \
            f"Shape mismatch: X={X_train.shape}, y={y_train.shape}"
        
        # 5. Refit GP model with new data
        gp = SingleTaskGP(
            torch.tensor(X_train, dtype=torch.double).to(device), 
            torch.tensor(y_train, dtype=torch.double).unsqueeze(-1).to(device)) # Unsqueeze for correct shape
        mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
        fit_gpytorch_mll(mll)

        # 6. Early stopping check
        if best_energy < energy_next + 1e-9:
            best_energy = energy_next
            patience_counter += 1

        if patience_counter >= PATIENCE:
            print(f"Early stopping at iteration {iteration+1} due to no improvement in energy for {PATIENCE} iterations.")
            break
        
        print(f"Iteration {iteration+1}/{MAX_ITER}, R: {R_next:.4f}, Energy: {np.exp(energy_next):.4f}")
    
    # Step 4: Final results
    best_index = np.argmin(-y_train) # Get index of best energy (negate back)
    best_R = X_train[best_index, 0]
    best_centers = X_train[best_index, 1:3*K+1].reshape(K, 3)
    #best_centers = denormalize(best_centers) # Denormalize centers to original scale
    best_normals = X_train[best_index, 3*K+1:].reshape(K, 2)
    #best_normals = denormalize(best_normals) # Denormalize normals to original scale
    # Create folder if it doesn't exist and print to csv
    os.makedirs("configurations", exist_ok=True)
    with open(f"configurations/best_configuration_K{K}_{SEED}.csv", "w") as f:
        f.write(f"R,cX,cY,cZ,nX,nY,nZ\n")
        for i in range(K):
            f.write(f"{best_R},")
            f.write(f"{','.join(map(str, spherical_to_cartesian(best_centers[i, 0], best_centers[i, 1], best_centers[i, 2]).flatten()))},")
            f.write(f"{','.join(map(str, spherical_to_cartesian(np.ones(1), best_normals[i, 0], best_normals[i, 1]).flatten()))}\n")
        print(f"Best configuration saved to configurations/best_configuration_K{K}_{SEED}.csv")


if __name__ == "__main__":    main()