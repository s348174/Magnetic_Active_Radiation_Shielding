# Import libraries for optimization
import torch
# BoTorch imports
from botorch.models import SingleTaskGP # Gaussian Process model for single-task regression
from botorch.models.transforms.outcome import Standardize # Outcome transform to standardize targets
from botorch.fit import fit_gpytorch_mll # Model fitting utility
from botorch.acquisition import qLogNoisyExpectedImprovement, PosteriorMean, qKnowledgeGradient # Acquisition functions for BO
from botorch.optim.initializers import gen_one_shot_kg_initial_conditions
from botorch.optim import optimize_acqf # Optimization utility for acquisition functions

# GPyTorch imports
from gpytorch.mlls import ExactMarginalLogLikelihood # Marginal log likelihood for GP fitting
from gpytorch.kernels import MaternKernel, ScaleKernel # Kernels for GP (Matern with ARD + scaling)

# Import standard libraries
import numpy as np
import warnings
from joblib import Parallel, delayed

# Import externals functions
from BoUtils import (
    denormalize,
    unpack_configuration,
    map_fn,
    InputMappedAcquisition,
    InputMappedModel,
    sobol_sample,
    stopping_criterion,
    duplicate_safe_candidate,
    fit_gp_with_noise_floor
)
from Objective import objective_function
from input import device, MAX_ITER, unit_bounds, MAPPED_D, rng, Q


def _likelihood_noise_scalar(gp):
    noise = torch.as_tensor(gp.likelihood.noise, dtype=torch.double, device=device)
    return noise.mean().item()

def bo_matern_kernel(nu=2.5):
    warnings_counter = 0
    # Step 1: Initialize BO with Sobol samples
    X_train, y_train, train_yvar = sobol_sample()
    train_yvar = torch.tensor(train_yvar, dtype=torch.double).unsqueeze(-1).to(device) # Convert to tensor for noise floor fitting
    train_yvar = torch.clamp(train_yvar, min=1e-6) # Avoid zero noise variance which can cause GP fitting issues

    ## Step 2: Fit Gaussian Process model over centers and normals
    covar_module = ScaleKernel(
        MaternKernel(
            nu=nu, # Smoothness parameter for Matern kernel (common choice for BO)
            ard_num_dims=MAPPED_D, # Must match the feature-mapped input dimension
        )
    )
    X_train_model = map_fn(X_train)
    gp = SingleTaskGP(
        X_train_model.detach().clone().to(device),
        torch.tensor(y_train, dtype=torch.double).unsqueeze(-1).to(device), # Unsqueeze for correct shape
        covar_module=covar_module,
        outcome_transform=Standardize(m=1),
        train_Yvar=train_yvar, # Heteroskedastic noise variance from initial evaluations
    )
    fit_gp_with_noise_floor(gp)

    # Initialize empty list of best means and ei values
    ei_array = np.empty(0)
    best_mean_hist = []
    variance_hist = []
    # Step 3: BO loop
    for iteration in range(MAX_ITER):
        # 1. Optimize acquisition function to find next point
        qKG = qKnowledgeGradient(model=InputMappedModel(gp, map_fn), num_fantasies=2) # More fantasies can give better performance but increase runtime
        candidate, qkg_value = optimize_acqf(
            acq_function=qKG,
            bounds=unit_bounds,
            q=Q,
            num_restarts=40,
            raw_samples=512,
            ic_generator=gen_one_shot_kg_initial_conditions,
        )
        # Check for duplicates in training data (can happen due to optimization tolerances)
        cand = candidate[0:1].detach().clone().to(device)  # Take only first from batch
        if duplicate_safe_candidate(X_train, cand):
            print("Duplicate candidate returned by optimizer — skipping this iteration")
            continue
        
        # 2. Extract params from candidate and generate random centers and normals
        candidate_np = candidate.detach().cpu().numpy()
        candidate_unnorm = denormalize(candidate_np) # Denormalize candidate to original scale
        centers_next, normals_next = unpack_configuration(candidate_unnorm[0])
        
        # 3. Evaluate simulation at new point (parallelized)
        energy_next, var_next = objective_function(rng.integers(1e6), centers_next, normals_next)                
        train_yvar = torch.cat((train_yvar, torch.tensor([[var_next]], dtype=torch.double).to(device)), dim=0) # Append new variance to training noise variances
        train_yvar = torch.clamp(train_yvar, min=1e-6) # Ensure noise variance doesn't go to zero 
        if np.isnan(energy_next) or np.isinf(energy_next):
            warnings.warn(f"Non-finite energy from simulator encountered at iteration {iteration+1}. Skipping.")
            warnings_counter += 1
            if warnings_counter > MAX_ITER // 3:
                raise ValueError("Too many non-finite energies encountered.")
            continue  # Skip this iteration and try again with a new candidate
        
        # 4. Update training data (skip duplicates)
        X_train = torch.vstack((X_train, cand)) # candidate is already normalized
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
            train_Yvar=train_yvar,
        )
        fit_gp_with_noise_floor(gp)

        # 6. Check convergence via best predicted value
        max_ei = qkg_value.item()
        ei_array = np.append(ei_array, max_ei)
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
        learned_noise_var = _likelihood_noise_scalar(gp)
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
        variance_hist.append(posterior_var)
        if stopping_criterion(best_mean_hist, learned_noise_var, posterior_var):
            print(f"Convergence reached at iteration {iteration+1} with EI={np.exp(max_ei):.6f}, predicted mean={current_best_mean:.4f}, and variance={posterior_var:.6f}")
            break
        print(f"Iteration {iteration+1}/{MAX_ITER}, Energy: {np.exp(energy_next):.4f}")
    print(f"BO loop completed with {warnings_counter} non-finite energy warnings.")
    return X_train.cpu().numpy(), y_train, ei_array, best_mean_hist, variance_hist

def bo_rbf_kernel():
    warnings_counter = 0
    # Step 1: Initialize BO with Sobol samples
    X_train, y_train, _ = sobol_sample()

    # Step 2: Fit GP model
    gp = SingleTaskGP(
        X_train.detach().clone().to(device),
        torch.tensor(y_train, dtype=torch.double).unsqueeze(-1).to(device), # Unsqueeze for correct shape
    )
    mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
    fit_gpytorch_mll(mll)

    # Initialize empty list of best means and ei values
    ei_array = np.empty(0)
    best_mean_hist = []
    variance_hist = []
    # Step 3: BO loop
    for iteration in range(MAX_ITER):
        # 1. Optimize acquisition function to find next point
        #ei = LogExpectedImprovement(gp, best_f=torch.tensor(y_train.max(), dtype=torch.double).to(device))
        qLogNEI = qLogNoisyExpectedImprovement(
            model=gp, 
            X_baseline=X_train,
            )
        candidate, _ = optimize_acqf(
            acq_function=qLogNEI,
            bounds=unit_bounds,
            q=Q,
            num_restarts=60,
            raw_samples=1024,
        )
        # Check for duplicates in training data (can happen due to optimization tolerances)
        cand = candidate[0:1].detach().clone().to(device)  # Take only first from batch
        if duplicate_safe_candidate(X_train, cand):
            print("Duplicate candidate returned by optimizer — skipping this iteration")
            continue
        
        # 2. Extract params from candidate and generate random centers and normals
        candidate_np = candidate.detach().cpu().numpy()
        candidate_unnorm = denormalize(candidate_np) # Denormalize candidate to original scale
        centers_next, normals_next = unpack_configuration(candidate_unnorm[0])
        
        # 3. Evaluate simulation at new point
        energy_next = objective_function(rng.integers(1e6), centers_next, normals_next)
        if np.isnan(energy_next) or np.isinf(energy_next):
            warnings.warn(f"Non-finite energy from simulator encountered at iteration {iteration+1}. Skipping.")
            warnings_counter += 1
            if warnings_counter >= MAX_ITER // 3:
                raise ValueError("Too many non-finite energies encountered.")
            continue  # Skip this iteration and try again with a new candidate
        
        # 4. Update training data
        X_train = torch.vstack((X_train, candidate[0:1])) # candidate is already normalized, take only first from batch
        y_train = np.append(y_train, float(-energy_next)) # Negate energy for maximization
        # Check shapes before fitting GP
        assert X_train.shape[0] == y_train.shape[0], \
            f"Shape mismatch: X={X_train.shape}, y={y_train.shape}"
        
        # 5. Refit GP model with new data
        gp = SingleTaskGP(
            X_train.detach().clone().to(device),
            torch.tensor(y_train, dtype=torch.double).unsqueeze(-1).to(device), # Unsqueeze for correct shape
        )
        mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
        fit_gpytorch_mll(mll)

        # 6. Check convergence via best predicted value
        max_ei = qLogNEI(candidate).item()
        ei_array = np.append(ei_array, np.exp(max_ei))
        post_mean = PosteriorMean(gp)
        _, current_best_mean = optimize_acqf(
            acq_function=post_mean,
            bounds=unit_bounds,
            q=1,
            num_restarts=60,
            raw_samples=1024,
            )
        best_mean_hist.append(current_best_mean)
        learned_noise_var = _likelihood_noise_scalar(gp)
        # Use top-5 observed points — more stable than a single noisy incumbent
        top_k = min(5, X_train.shape[0])
        top_idx = torch.as_tensor(
            np.argsort(y_train)[-top_k:],
            dtype=torch.long,
            device=device,
        )
        eval_X = X_train[top_idx]
        posterior = gp.posterior(eval_X)
        posterior_var = posterior.variance.mean().item()
        variance_hist.append(posterior_var)
        if stopping_criterion(best_mean_hist, learned_noise_var, posterior_var):
            print(f"Convergence reached at iteration {iteration+1} with EI={np.exp(max_ei):.6f}, predicted mean={current_best_mean:.4f}, and variance={posterior_var:.6f}")
            break
        print(f"Iteration {iteration+1}/{MAX_ITER}, Energy: {np.exp(energy_next):.4f}")
    print(f"BO loop completed with {warnings_counter} non-finite energy warnings.")
    return X_train.cpu().numpy(), y_train, ei_array, best_mean_hist, variance_hist
