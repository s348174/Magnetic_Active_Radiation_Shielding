# Import libraries for optimization
import torch
from botorch.acquisition import AcquisitionFunction
from botorch.utils.sampling import draw_sobol_samples # Better sampling startegy in high dimensions than random sampling

# Import standard libraries
import numpy as np

# Import externals functions
from Objective import objective_function

# Import imput variables
from input import K, device, LOWER, UPPER, D, USE_FEATURE_MAPPING, PERIODIC_IDXS, unit_bounds, rng, INIT, SEED, CONVERGENCE_THRESHOLD, Q

###################################################################
# NORMALIZATION FUNCTIONS AND CONFIGURATION PACKING/UNPACKING
###################################################################
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

########################################################
# CHANGE OF FEATURES FOR PERIODIC VARIABLES
########################################################
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
    

# FUNCTIONS FOR BO LOOP
def sobol_sample():
    # Step 1: Initialize BO with Sobol samples
    X_train = draw_sobol_samples(unit_bounds, n=INIT, q=Q, seed=SEED).squeeze(1).to(device) # shape: (INIT, D)
    X_train_unnorm = denormalize(X_train).cpu().numpy() # shape: (INIT, D)
    centers_init = np.empty((INIT, K, 3))
    normals_init = np.empty((INIT, K, 2))
    for i in range(INIT):
        centers_init[i], normals_init[i] = unpack_configuration(X_train_unnorm[i])
    energy_init = np.array([
        objective_function(rng.integers(1e6), centers_init[i], normals_init[i])
        for i in range(INIT)
    ]) # shape: (INIT,)
    y_train = -energy_init.flatten() # Negate energy for maximization
    return X_train, y_train

def stopping_criterion(best_mean_hist, learned_noise_var, posterior_var):
    if len(best_mean_hist) > 2*5*K: # Check convergence only after enough iterations to have a history
        # Compute cumulative difference of last 10 best means and check if it's below threshold relative to the value 10 iterations ago
        cumulative_difference = sum(abs(best_mean_hist[-i] - best_mean_hist[-i-1]) for i in range(1, 11)) / 10.0
        if (cumulative_difference < CONVERGENCE_THRESHOLD * abs(best_mean_hist[-10])):
            print(f"Convergence reached with best mean criterion")
            return True

        if posterior_var < CONVERGENCE_THRESHOLD * max(learned_noise_var, 1e-10):
            print(f"Convergence reached with posterior variance criterion")
            return True
    return False