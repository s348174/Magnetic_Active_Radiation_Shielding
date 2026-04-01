# Import C++ bindings
import sys
sys.path.append('../Release')
import simulator

# Import libraries for optimization
import torch
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_mll
from botorch.acquisition import ExpectedImprovement
from botorch.optim import optimize_acqf
from gpytorch.mlls import ExactMarginalLogLikelihood

# Import standard libraries
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# PARAMETERS SETUP
BOUNDS = (2.0, 5.0)
K = 5 # Number of coils
I = 1E4 # Current in Amperes
R_BOUNDS = (0.1, 2.0) # Coil radius bounds in meters
MAX_ITER = 100
SEED = 42
rng = np.random.default_rng(SEED)


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
    return expected_energy

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

def initialize_bo(n_initial_points=K):
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

initialize_bo()