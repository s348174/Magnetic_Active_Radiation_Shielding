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
import random as rng
import os
import csv
import time

# Import externals functions
from LogDataSampler import sample_all_particles
from PlotResults import interactive_plot
from BoUtils import spherical_to_cartesian, unpack_configuration, map_fn, denormalize, InputMappedAcquisition

from optimizer_py.input import K, N, I, R, device, MAX_ITER, CONVERGENCE_THRESHOLD, INIT, unit_bounds, SEED, D, LOG_DATA, MAPPED_D

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