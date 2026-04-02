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

# FIELD HYPERPARAMETERS SETUP
BOUNDS = (-1.0, 1.0)
K = 1 # Number of coils
I = 1E5 # Current in Amperes
R_BOUNDS = (0.1, 2.0) # Coil radius bounds in meters

# SIMULATION AND OPTIMIZATION HYPERPARAMETERS
INIT = 1 # Number of initial random samples for BO
MAX_ITER = 10
PATIENCE = 5 # Patience for early stopping
SEED = 42
rng = np.random.default_rng(SEED)

# Set device for PyTorch (use GPU if available)
device = torch.device("cpu")
print(f"Using device: {device}")


def objective_function(seed, R, centers, normals):
    """
    Calls the C++ simulator to compute the energy hitting the detector.
    R: is coils radius
    centers: list of coil centers
    normals: list of coil normal vectors
    Returns: energy hitting the detector (lower is better)
    """
    # Call the C++ simulator
    expected_energy = simulator.launch_simulation(seed, K, I, R, centers, normals)
    # Take the log of energy to stabilize optimization and handle wide range of values
    return np.log(expected_energy + 1e-8) # Add small value to avoid log(0)

def random_coil_configuration(K):
    """
    Generates random coil configuration (centers and normals).
    """
    centers = np.array([]).reshape(0, 3) # Initialize empty array for centers
    for i in range(K):
        center = rng.uniform(BOUNDS[0], BOUNDS[1], size=3) # Random centers in 3D
        centers = np.append(centers, [center], axis=0) # Append to centers array
    print("Generated random coil configuration:")
    print("Centers:\n", centers)
    normals = np.array([]).reshape(0, 3) # Initialize empty array for normals
    for i in range(K):
        normal = rng.normal(0, 1, size=3) # Random normal in 3D
        normal = normal / np.linalg.norm(normal) # Normalize
        normals = np.append(normals, [normal], axis=0) # Append to normals array
    print("Normals:\n", normals)    
    R = rng.uniform(R_BOUNDS[0], R_BOUNDS[1]) # Random coil radius
    return R, np.array(centers), np.array(normals)

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
    X_train = np.hstack((X_train, normals_init.reshape(-1, 3*K)))
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
            bounds=torch.tensor([
            [R_BOUNDS[0]] + [BOUNDS[0]]*3*K + [0]*3*K,
            [R_BOUNDS[1]] + [BOUNDS[1]]*3*K + [1]*3*K
            ], dtype=torch.double, device=device), 
            q=1,
            num_restarts=5,
            raw_samples=20,
        )
        
        # 2. Extract R from candidate and generate random centers and normals
        R_next = candidate[0, 0].item()
        centers_next = candidate[0, 1:3*K+1].reshape(K, 3).detach().numpy()
        normals_next = candidate[0, 3*K+1:].reshape(K, 3).detach().numpy()
        # Normalize normals
        normals_next = normals_next / np.linalg.norm(normals_next, axis=1, keepdims=True)
        
        # 3. Evaluate simulation at new point
        energy_next = objective_function(rng.integers(1e6), R_next, centers_next, normals_next)
        
        # 4. Update training data
        X_train = np.vstack((X_train, np.hstack((R_next, centers_next.flatten(), normals_next.flatten()))))
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
    best_normals = X_train[best_index, 3*K+1:].reshape(K, 3)
    # Create folder if it doesn't exist and print to csv
    os.makedirs("configurations", exist_ok=True)
    with open(f"configurations/best_configuration_{SEED}.csv", "w") as f:
        f.write(f"R,cX,cY,cZ,nX,nY,nZ\n")
        f.write(f"{best_R},")
        f.write(f"{','.join(map(str, best_centers.flatten()))},")
        f.write(f"{','.join(map(str, best_normals.flatten()))}\n")
        print(f"Best configuration saved to configurations/best_configuration_{SEED}.csv")


if __name__ == "__main__":    main()