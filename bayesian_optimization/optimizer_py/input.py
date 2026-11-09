import torch
import numpy as np
import pandas as pd
import os

# Set device for PyTorch (use GPU if available)
device = torch.device("cpu")

# Field parameters (fixed for this optimization)
K = 10 # Number of coils
N = int(1e3) # Number of particles
ENERGY_BINS = 8 # Logarithmic energy strata; at least two samples per nonempty bin
I = 7.2E4 # Current in Amperes
R = 0.25 # Coil radius in meters

# SIMULATION AND OPTIMIZATION HYPERPARAMETERS
D = 5 * K # Input dimension (5 parameters per coil: r, theta, phi for center and theta, phi for normal)
INIT = 5 * D # Number of initial random samples for BO
MAX_ITER = 25 * D # Maximum number of BO iterations
CONVERGENCE_THRESHOLD = 1e-4 # Threshold for convergence
Q = 1 # Batch size: this pipeline optimizes one configuration at a time
SEED = 67
rng = np.random.default_rng(SEED)

# Multiple independent trust-region runs share the initial design and MAX_ITER budget.
TURBO_REGIONS = 0       # 0: automatic effective-rank choice; set 4 or 5 to fix the count
TURBO_MAX_REGIONS = 5
FINAL_PARTICLES = int(1e5) # Per species, in each independent validation Monte Carlo
FINAL_REPEATS = 3
FINALISTS = 5

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
unit_bounds = torch.zeros(2, D, dtype=torch.double, device=device)
unit_bounds[1] = 1.0  # upper bounds all 1

# Indices of periodic variables in normalized design x in [0, 1]^D.
# We map these dimensions to sin/cos pairs to avoid wrap-around discontinuities.
PERIODIC_IDXS = list(range(2 * K, 3 * K)) + list(range(4 * K, 5 * K))

# Optional periodic map for the global loops; Cartesian maps always convert physical geometry.
USE_FEATURE_MAPPING = os.getenv("USE_FEATURE_MAPPING", "0").lower() not in ("0", "false", "f", "no")

# Input dimension seen by the GP after feature mapping.
if USE_FEATURE_MAPPING:
    MAPPED_D = D + len(PERIODIC_IDXS)
else:
    MAPPED_D = D

# Physical energies stay in MeV/amu; logarithmic bins control tail coverage.
FILEPATH = "../data/flux_data_clean.csv"
FLUX_DATA = pd.read_csv(FILEPATH)
LOG_DATA = FLUX_DATA # Compatibility with existing analysis scripts
