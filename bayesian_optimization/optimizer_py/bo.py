# Import C++ bindings
import sys
from xml.parsers.expat import model
sys.path.append('../Release')
import simulator

# Import libraries for optimization
import torch
# BoTorch imports
from botorch.models import SingleTaskGP # Gausian Process model for single-task regression
from botorch.models.transforms.outcome import Standardize # Outcome transform to standardize targets
from botorch.fit import fit_gpytorch_mll # Model fitting utility
from botorch.acquisition import LogExpectedImprovement, qLogNoisyExpectedImprovement, PosteriorMean # Acquisition functions for BO
from botorch.acquisition.acquisition import AcquisitionFunction
from botorch.optim import optimize_acqf # Optimization utility for acquisition functions
from botorch.utils.sampling import draw_sobol_samples # Better sampling startegy in high dimensions than random sampling
# GPyTorch imports
from gpytorch.mlls import ExactMarginalLogLikelihood # Marginal log likelihood for GP fitting
from gpytorch.kernels import MaternKernel, ScaleKernel # Kernels for GP (Matern with ARD + scaling)

# Import standard libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm
import os
import csv
import time

# Import externals functions
from LogDataSampler import sample_all_particles
from PlotResults import interactive_plot

# Set device for PyTorch (use GPU if available)
device = torch.device("cpu")
print(f"Using device: {device}")

# Field parameters (fixed for this optimization)
K = 4 # Number of coils
N = int(1e3) # Number of particles
I = 7.2E4 # Current in Amperes
R = 0.5 # Initial coil radius in meters

# SIMULATION AND OPTIMIZATION HYPERPARAMETERS
D = 5 * K # Input dimension (5 parameters per coil: r, theta, phi for center and theta, phi for normal)
INIT = 5*D # Number of initial random samples for BO
MAX_ITER = 1000 # Maximum number of BO iterations
CONVERGENCE_THRESHOLD = 1e-6 # Threshold for convergence
SEED = 67
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
unit_bounds = torch.zeros(2, D, dtype=torch.double, device=device)
unit_bounds[1] = 1.0  # upper bounds all 1

# Indices of periodic variables in normalized design x in [0, 1]^D.
# We map these dimensions to sin/cos pairs to avoid wrap-around discontinuities.
PERIODIC_IDXS = list(range(2 * K, 3 * K)) + list(range(4 * K, 5 * K))

# Quick toggle to enable/disable periodic feature mapping.
# Set environment variable `USE_FEATURE_MAPPING=0` or `USE_FEATURE_MAPPING=false` to disable.
os.environ["USE_FEATURE_MAPPING"] = "1" # Default to enabled for better performance, but can be turned off for testing
USE_FEATURE_MAPPING = os.getenv("USE_FEATURE_MAPPING", "1").lower() not in ("0", "false", "f", "no")

# Input dimension seen by the GP after feature mapping.
if USE_FEATURE_MAPPING:
    MAPPED_D = D + len(PERIODIC_IDXS)
else:
    MAPPED_D = D

# Read initial data from log files
FILEPATH = "../data/log_scaled_flux_data.csv"
LOG_DATA = pd.read_csv(FILEPATH)

# Normalization functions to scale parameters to [0, 1] for optimization
def normalize(X):
    """Scale X from original bounds to [0, 1]."""
    # Move to numpy to avoid torch issues
    UPPER_np = UPPER.cpu().numpy()
    LOWER_np = LOWER.cpu().numpy()
    scaled = (X - LOWER_np) / (UPPER_np - LOWER_np)
    scaled = np.clip(scaled, 0.0, 1.0)
    return torch.tensor(scaled, dtype=torch.double).to(device)

def denormalize(X):
    """Scale X from [0, 1] back to original bounds."""
    # Move to numpy to avoid torch issues
    UPPER_np = UPPER.cpu().numpy()
    LOWER_np = LOWER.cpu().numpy()
    scaled = X * (UPPER_np - LOWER_np) + LOWER_np
    return torch.tensor(scaled, dtype=torch.double).to(device)

def pack_configuration(centers, normals):
    """Pack one sample with ordering consistent with LOWER/UPPER tensors."""
    return np.hstack([
        centers[:, 0],  # all center radii
        centers[:, 1],  # all center theta
        centers[:, 2],  # all center phi
        normals[:, 0],  # all normal theta
        normals[:, 1],  # all normal phi
    ])

def unpack_configuration(x):
    """Unpack one sample from LOWER/UPPER-consistent ordering."""
    center_r = x[0:K]
    center_theta = x[K:2*K]
    center_phi = x[2*K:3*K]
    normal_theta = x[3*K:4*K]
    normal_phi = x[4*K:5*K]
    centers = np.stack([center_r, center_theta, center_phi], axis=1)
    normals = np.stack([normal_theta, normal_phi], axis=1)
    return centers, normals

def periodic_feature_map(X):
    """
    Map periodic components x in [0, 1] to sin/cos pairs.
    Non-periodic components are kept unchanged.

    Input shape:  (..., D)
    Output shape: (..., D + len(PERIODIC_IDXS))
    """
    if not torch.is_tensor(X):
        X = torch.tensor(X, dtype=torch.double, device=device)
    X = X.to(dtype=torch.double, device=device)

    features = []
    periodic_set = set(PERIODIC_IDXS)
    for d in range(D):
        xd = X[..., d:d+1]
        if d in periodic_set:
            angle = 2.0 * np.pi * xd
            features.append(torch.sin(angle))
            features.append(torch.cos(angle))
        else:
            features.append(xd)
    return torch.cat(features, dim=-1)


def identity_map(X):
    """Return inputs unchanged (but ensure correct dtype/device).

    This is used when feature mapping is disabled so the rest of the
    code can call `map_fn(X)` uniformly.
    """
    if not torch.is_tensor(X):
        X = torch.tensor(X, dtype=torch.double, device=device)
    return X.to(dtype=torch.double, device=device)

# Choose mapping function depending on the toggle
map_fn = periodic_feature_map if USE_FEATURE_MAPPING else identity_map

class InputMappedAcquisition(AcquisitionFunction):
    """Evaluate an acquisition on mapped inputs while optimizing in original space."""

    def __init__(self, base_acqf, map_fn):
        super().__init__(model=base_acqf.model)
        self.base_acqf = base_acqf
        self.map_fn = map_fn

    def forward(self, X):
        return self.base_acqf(self.map_fn(X))

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
    r = np.asarray(r)
    theta = np.asarray(theta)
    phi = np.asarray(phi)
    x = r * np.sin(phi) * np.cos(theta)
    y = r * np.sin(phi) * np.sin(theta)
    z = r * np.cos(phi)
    return np.stack([x, y, z], axis=-1)

def objective_function(seed, centers, normals):
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
    
    # Sample particle energies from the log data distribution (for all particles, but we can choose one if needed)
    samples_dict = sample_all_particles(LOG_DATA, n_samples=N)
    
    # Call the C++ simulator
    expected_energy = simulator.launch_simulation(seed, N, K, I, R, centers_cart, normals_cart, samples_dict)
    # Take the log of energy to stabilize optimization and handle wide range of values
    log_energy = np.log(expected_energy)
    if np.isnan(log_energy) or np.isinf(log_energy):
        print(f"Warning: Simulator returned invalid energy {expected_energy} for seed {seed}. Retrying with a new seed.")
        seed = rng.integers(1e6)
        return objective_function(seed, centers, normals)
    return log_energy

def random_coil_configuration():
    """
    Generates random coil configuration in spherical coordinates.
    """
    # --- Centers (sampled in spherical, returned in cartesian) ---
    center_r     = rng.uniform(X_BOUNDS[0],     X_BOUNDS[1],     size=K)
    center_theta = rng.uniform(THETA_BOUNDS[0],  THETA_BOUNDS[1], size=K)
    center_phi   = rng.uniform(PHI_BOUNDS[0],    PHI_BOUNDS[1],   size=K)
    centers = np.column_stack((center_r, center_theta, center_phi))
    # --- Normals (unit vectors — r=1, sampled in spherical, returned in cartesian) ---
    normal_theta = rng.uniform(THETA_BOUNDS[0], THETA_BOUNDS[1], size=K)
    normal_phi   = rng.uniform(PHI_BOUNDS[0],   PHI_BOUNDS[1],   size=K)
    normals = np.column_stack((normal_theta, normal_phi))
    return centers, normals

def initialize_random_bo(n_initial_points=INIT):
    """
    Initializes the Bayesian Optimization process with random samples.
    Returns initial data (centers, normals, energy).
    """
    centers_samples = []
    normals_samples = []
    energy_samples = []
    
    for _ in range(n_initial_points):
        centers, normals = random_coil_configuration()
        energy = objective_function(rng.integers(1e6), centers, normals)

        centers_samples.append(centers)
        normals_samples.append(normals)
        energy_samples.append(energy)
    
    return np.array(centers_samples), np.array(normals_samples), np.array(energy_samples)

def main():
    start_clock = time.time()

    ei_array = np.empty(0)

    # Step 1: Initialize BO with random samples
    #centers_init, normals_init, energy_init = initialize_random_bo()
    X_train = draw_sobol_samples(unit_bounds, n=INIT, q=1, seed=SEED).squeeze(1).to(device) # shape: (INIT, D)
    X_train_unnorm = denormalize(X_train).cpu().numpy() # shape: (INIT, D)
    centers_init = np.empty((INIT, K, 3))
    normals_init = np.empty((INIT, K, 2))
    for i in range(INIT):
        centers_init[i], normals_init[i] = unpack_configuration(X_train_unnorm[i])
    energy_init = np.array([
        objective_function(rng.integers(1e6), centers_init[i], normals_init[i])
        for i in range(INIT)
    ]) # shape: (INIT,)

    ## Step 2: Fit Gaussian Process model over centers and normals
    """X_train_unnorm = np.array([
        pack_configuration(centers_init[i], normals_init[i])
        for i in range((centers_init).shape[0])
    ])"""
    #X_train = normalize(X_train_unnorm) # Keep X_train normalized throughout
    y_train = -energy_init.flatten() # Negate energy for maximization

    # Fit GP model
    covar_module = ScaleKernel(
        MaternKernel(
            nu=2.5, # Smoothness parameter for Matern kernel (common choice for BO)
            ard_num_dims=MAPPED_D, # Must match the feature-mapped input dimension
        )
    )
    X_train_model = map_fn(X_train)
    gp = SingleTaskGP(
        X_train_model.detach().clone().to(device),
        torch.tensor(y_train, dtype=torch.double).unsqueeze(-1).to(device), # Unsqueeze for correct shape
        covar_module=covar_module,
        outcome_transform=Standardize(m=1),
    )
    mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
    fit_gpytorch_mll(mll)

    # Initialize empty list of best means
    best_mean_hist = []
    # Step 3: BO loop
    for iteration in range(MAX_ITER):
        # 1. Optimize acquisition function to find next point
        #ei = LogExpectedImprovement(gp, best_f=torch.tensor(y_train.max(), dtype=torch.double).to(device))
        ei = qLogNoisyExpectedImprovement(
            model=gp, 
            X_baseline=X_train_model,
            )
        ei_on_original_space = InputMappedAcquisition(ei, map_fn)
        candidate, _ = optimize_acqf(
            acq_function=ei_on_original_space,
            bounds=unit_bounds,
            q=1,
            num_restarts=60,
            raw_samples=1024,
        )

        # Convergence with EI threshold
        max_ei = ei_on_original_space(candidate).item()
        ei_array = np.append(ei_array, np.exp(max_ei))
        """best_energy_so_far = -y_train.max()
        if max_ei < np.log(CONVERGENCE_EI_THRESHOLD) + best_energy_so_far:
            # If expected improvement is very small, we can stop
            print(f"Convergence reached at iteration {iteration+1} with EI={np.exp(max_ei):.6f}")
            break  """
        
        # 2. Extract params from candidate and generate random centers and normals
        candidate_np = candidate.detach().cpu().numpy()
        candidate_unnorm = denormalize(candidate_np) # Denormalize candidate to original scale
        centers_next, normals_next = unpack_configuration(candidate_unnorm[0])
        
        # 3. Evaluate simulation at new point
        energy_next = objective_function(rng.integers(1e6), centers_next, normals_next)
        
        # 4. Update training data
        X_train = torch.vstack((X_train, candidate.reshape(1, -1))) # candidate is already normalized
        y_train = np.append(y_train, float(-energy_next)) # Negate energy for maximization
        # Check shapes before fitting GP
        assert X_train.shape[0] == y_train.shape[0], \
            f"Shape mismatch: X={X_train.shape}, y={y_train.shape}"
        
        # 5. Refit GP model with new data
        X_train_model = map_fn(X_train)
        gp = SingleTaskGP(
            X_train_model.detach().clone().to(device),
            torch.tensor(y_train, dtype=torch.double).unsqueeze(-1).to(device), # Unsqueeze for correct shape
            covar_module=covar_module,
            outcome_transform=Standardize(m=1),
        )
        mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
        fit_gpytorch_mll(mll)

        # 6. Check convergence via best predicted value
        post_mean = PosteriorMean(gp)
        post_mean_on_original_space = InputMappedAcquisition(post_mean, map_fn)
        _, current_best_mean = optimize_acqf(
            acq_function=post_mean_on_original_space,
            bounds=unit_bounds,
            q=1,
            num_restarts=60,
            raw_samples=1024,
            )
        best_mean_hist.append(current_best_mean)
        if len(best_mean_hist) > 2*D: # Check convergence only after enough iterations to have a history
            # Compute cumulative difference of last 10 best means and check if it's below threshold relative to the value 10 iterations ago
            cumulative_difference = sum(abs(best_mean_hist[-i] - best_mean_hist[-i-1]) for i in range(1, 11)) / 10.0
            if (cumulative_difference < CONVERGENCE_THRESHOLD * abs(best_mean_hist[-10])):
                print(f"Convergence reached at iteration {iteration} with EI={np.exp(max_ei):.6f} and predicted mean={current_best_mean:.4f}")
                break

            learned_noise_var = gp.likelihood.noise.item()

            # Use top-5 observed points — more stable than a single noisy incumbent
            top_k = min(5, X_train.shape[0])
            top_idx = torch.as_tensor(
                np.argsort(y_train)[-top_k:],
                dtype=torch.long,
                device=device,
            )
            eval_X = X_train_model[top_idx]

            posterior = gp.posterior(eval_X)
            posterior_var = posterior.variance.mean().item()

            if posterior_var < CONVERGENCE_THRESHOLD * max(learned_noise_var, 1e-10):
                print(f"Convergence reached at iteration {iteration} with EI={np.exp(max_ei):.6f}, predicted mean={current_best_mean:.4f}, and variance={posterior_var:.6f}")
                break
        
        print(f"Iteration {iteration+1}/{MAX_ITER}, Energy: {np.exp(energy_next):.4f}")
    
    end_clock = time.time() - start_clock
    print(f'Elapsed time: {end_clock:.4f} s')

    # Step 4: Final results
    best_index = np.argmin(-y_train) # Get index of best energy (negate back)
    best_X_norm = X_train[best_index].detach().cpu().numpy()
    best_X = denormalize(best_X_norm[np.newaxis, :]).detach().cpu().numpy()[0]
    best_centers, best_normals = unpack_configuration(best_X)
    # Create folder if it doesn't exist and print to csv
    os.makedirs("configurations", exist_ok=True)
    with open(f"configurations/best_configuration_K{K}_{SEED}.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["cX", "cY", "cZ", "nX", "nY", "nZ"])
        for i in range(K):
            center_xyz = spherical_to_cartesian(best_centers[i, 0], best_centers[i, 1], best_centers[i, 2])
            normal_xyz = spherical_to_cartesian(1.0, best_normals[i, 0], best_normals[i, 1])
            center_xyz = center_xyz[0] if np.ndim(center_xyz) > 1 else center_xyz
            normal_xyz = normal_xyz[0] if np.ndim(normal_xyz) > 1 else normal_xyz
            writer.writerow([center_xyz[0], center_xyz[1], center_xyz[2], normal_xyz[0], normal_xyz[1], normal_xyz[2]])
        print(f"Best configuration saved to configurations/best_configuration_K{K}_{SEED}.csv")
    
    plt.figure(figsize=(10, 6))
    plt.plot(ei_array, marker='o')
    plt.title("Expected Improvement over time")
    plt.xlabel("Iteration")
    plt.ylabel("Expected Improvement")
    plt.show()

    plt.figure(figsize=(10, 6))
    plt.plot(best_mean_hist, marker='o')
    plt.title("Best Predicted Mean over time")
    plt.xlabel("Iteration")
    plt.ylabel("Best Predicted Mean")
    plt.show()

    # Plot the best configuration
    interactive_plot(result_file=f"configurations/best_configuration_K{K}_{SEED}.csv")

if __name__ == "__main__":    main()