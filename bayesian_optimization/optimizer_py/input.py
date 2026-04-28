import torch
import numpy as np
import pandas as pd

# Set device for PyTorch (use GPU if available)
device = torch.device("cpu")
print(f"Using device: {device}")

# Field parameters (fixed for this optimization)
K = 4 # Number of coils
N = int(1e3) # Number of particles
I = 7.2E4 # Current in Amperes
R = 0.5 # Initial coil radius in meters

# SIMULATION AND OPTIMIZATION HYPERPARAMETERS
INIT = 2*5*K # Number of initial random samples for BO
MAX_ITER = 500 # Maximum number of BO iterations
CONVERGENCE_THRESHOLD = 1e-6 # Threshold for convergence
SEED = 42
rng = np.random.default_rng(SEED)

# FIELD HYPERPARAMETERS SETUP
X_BOUNDS = (1.0, 4.0) # Bound for distance of coil centers from origin in meters
THETA_BOUNDS = (0, np.pi) # Bounds for azimuthal angles in spherical coordinates
PHI_BOUNDS = (0, 2*np.pi) # Bounds for polar angles in spherical coordinates

# Define bounds tensor
LOWER = torch.tensor(
    [X_BOUNDS[0]]*K + [THETA_BOUNDS[0]]*K + [PHI_BOUNDS[0]]*K + [THETA_BOUNDS[0]]*K + [PHI_BOUNDS[0]]*K,
    dtype=torch.double
).to(device)
UPPER = torch.tensor(
    [X_BOUNDS[1]]*K + [THETA_BOUNDS[1]]*K + [PHI_BOUNDS[1]]*K + [THETA_BOUNDS[1]]*K + [PHI_BOUNDS[1]]*K,
    dtype=torch.double
).to(device)

# Define unitary bounds
D = 5 * K
unit_bounds = torch.zeros(2, D, dtype=torch.double, device=device)
unit_bounds[1] = 1.0  # upper bounds all 1

# Read initial data from log files
FILEPATH = "../data/log_scaled_flux_data.csv"
LOG_DATA = pd.read_csv(FILEPATH)