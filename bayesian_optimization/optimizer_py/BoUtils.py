# Import libraries for optimization
import torch
from botorch.acquisition import AcquisitionFunction, OneShotAcquisitionFunction
from botorch.models.model import Model
from botorch.utils.sampling import draw_sobol_samples # Better sampling startegy in high dimensions than random sampling
from botorch.fit import fit_gpytorch_mll
from gpytorch.constraints import GreaterThan
from gpytorch.mlls import ExactMarginalLogLikelihood # Marginal log likelihood for GP fitting

# Import standard libraries
import numpy as np
from joblib import Parallel, delayed

# Import externals functions
from Objective import objective_function

# Import imput variables
from input import COPY, K, device, LOWER, UPPER, D, USE_FEATURE_MAPPING, PERIODIC_IDXS, unit_bounds, rng, INIT, SEED, CONVERGENCE_THRESHOLD, Q

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


class InputMappedOneShotAcquisition(OneShotAcquisitionFunction):
    """Evaluate a one-shot acquisition on mapped inputs while preserving KG semantics."""

    def __init__(self, base_acqf, map_fn):
        super().__init__(model=base_acqf.model)
        self.base_acqf = base_acqf
        self.map_fn = map_fn
        self.objective = getattr(base_acqf, "objective", None)
        self.posterior_transform = getattr(base_acqf, "posterior_transform", None)
        self.sampler = getattr(base_acqf, "sampler", None)
        self.inner_sampler = getattr(base_acqf, "inner_sampler", None)
        self.current_value = getattr(base_acqf, "current_value", None)
        self.num_fantasies = getattr(base_acqf, "num_fantasies", None)
        self.X_pending = getattr(base_acqf, "X_pending", None)
        self.cost_aware_utility = getattr(base_acqf, "cost_aware_utility", None)
        self.cost_sampler = getattr(base_acqf, "cost_sampler", None)
        self.project = getattr(base_acqf, "project", None)
        self.expand = getattr(base_acqf, "expand", None)

    def forward(self, X):
        return self.base_acqf(self.map_fn(X))

    def get_augmented_q_batch_size(self, q):
        return self.base_acqf.get_augmented_q_batch_size(q)

    def extract_candidates(self, X_full):
        return self.base_acqf.extract_candidates(X_full)


class InputMappedModel(Model):
    """Wrap a BoTorch model so it accepts original-space inputs and maps them internally."""

    def __init__(self, base_model, map_fn):
        super().__init__()
        self.base_model = base_model
        self.map_fn = map_fn

    def posterior(self, X, output_indices=None, observation_noise=False, posterior_transform=None):
        return self.base_model.posterior(
            self.map_fn(X),
            output_indices=output_indices,
            observation_noise=observation_noise,
            posterior_transform=posterior_transform,
        )

    def fantasize(self, X, sampler, observation_noise=None, **kwargs):
        fantasy_model = self.base_model.fantasize(
            self.map_fn(X),
            sampler=sampler,
            observation_noise=observation_noise,
            **kwargs,
        )
        return InputMappedModel(fantasy_model, self.map_fn)

    def condition_on_observations(self, X, Y, **kwargs):
        conditioned_model = self.base_model.condition_on_observations(
            self.map_fn(X),
            Y,
            **kwargs,
        )
        return InputMappedModel(conditioned_model, self.map_fn)

    def transform_inputs(self, X, input_transform=None):
        return self.map_fn(X)

    @property
    def batch_shape(self):
        return self.base_model.batch_shape

    @property
    def num_outputs(self):
        return self.base_model.num_outputs

    def subset_output(self, idcs):
        return InputMappedModel(self.base_model.subset_output(idcs), self.map_fn)

    def __getattr__(self, name):
        return super().__getattr__(name)
    
##########################################################
# FUNCTIONS FOR BO LOOP
##########################################################
def sobol_sample():
    # Step 1: Initialize BO with Sobol samples
    X_train = draw_sobol_samples(unit_bounds, n=INIT, q=1, seed=SEED).squeeze(1).to(device) # shape: (INIT, D)
    X_train_unnorm = denormalize(X_train).cpu().numpy() # shape: (INIT, D)
    centers_init = np.empty((INIT, K, 3))
    normals_init = np.empty((INIT, K, 2))
    energies = np.empty((INIT, COPY))
    for i in range(INIT):
        print(f"Evaluating initial configuration {i+1}/{INIT}...")
        centers_init[i], normals_init[i] = unpack_configuration(X_train_unnorm[i])
        energies[i] = Parallel(n_jobs=-1)(
            delayed(objective_function)(rng.integers(1e6), centers_init[i], normals_init[i]) 
            for _ in range(COPY)
        )
    energy_init = np.array([np.mean(energies[i]) for i in range(INIT)])
    train_yvar = np.array([np.var(energies[i]) for i in range(INIT)])
    y_train = -energy_init.flatten() # Negate energy for maximization
    # Remove any non-finite or nan values that might arise from the simulator
    finite_mask = np.isfinite(y_train)
    X_train = X_train[finite_mask]
    y_train = y_train[finite_mask]
    train_yvar = train_yvar[finite_mask]
    return X_train, y_train, train_yvar

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

def duplicate_safe_candidate(X_train, candidate, tol=1e-8):
    """Return True when candidate is already present in X_train."""
    cand = candidate.detach().clone().to(device).reshape(1, -1)
    dists = torch.norm(X_train - cand, dim=1)
    return bool((dists < tol).any())


def fit_gp_with_noise_floor(gp, noise_floor=1e-8, retry_noise=1e-6):
    """Fit a GP while keeping a small positive noise floor for stability."""
    try:
        gp.likelihood.noise_covar.register_constraint("raw_noise", GreaterThan(noise_floor))
    except Exception:
        pass

    mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
    try:
        fit_gpytorch_mll(mll)
    except Exception as e:
        print(f"Warning: GP fit failed ({e}), retrying with initialized noise")
        try:
            gp.likelihood.noise_covar.initialize(noise=retry_noise)
        except Exception:
            pass
        mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
        fit_gpytorch_mll(mll)