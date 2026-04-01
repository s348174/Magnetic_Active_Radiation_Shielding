# Import C++ bindings
import sys
sys.path.append('Release')
import simulator

# Import libraries for optimization
import torch
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_model
from botorch.acquisition import ExpectedImprovement
from botorch.optim import optimize_acqf
from gpytorch.mlls import ExactMarginalLogLikelihood

# Import standard libraries
import numpy as np
import pandas as pd
import matplotlib as plt
import sklearn as sk
import time
import random

# GPU device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------------------------
# CONFIGURATION
# -------------------------
K = 4                      # Number of coils
DIM = 6 * K                # Centers (3K) + Normals (3K)
BOUNDS = torch.tensor([
    [-10.0] * DIM,         # Lower bounds
    [10.0] * DIM           # Upper bounds
], dtype=torch.double)
I = 1e4;                   # Current intensity
R = 15;                    # Coil radius

SEED = 123


# -------------------------
# HELPER: decode parameters
# -------------------------
def decode_params(x: torch.Tensor):
    """
    Convert flat tensor -> centers + normals
    """
    x = x.view(-1)

    centers = x[:3*K].reshape(K, 3)
    normals = x[3*K:].reshape(K, 3)

    # normalize normals
    normals = normals / normals.norm(dim=1, keepdim=True)

    return centers, normals


# -------------------------
# OBJECTIVE FUNCTION
# -------------------------
def objective(x: torch.Tensor):
    """
    x: (batch, DIM)
    returns: (batch, 1)
    """
    results = []

    for xi in x:
        centers, normals = decode_params(xi)

        # convert to numpy
        centers_np = centers.detach().cpu().numpy()
        normals_np = normals.detach().cpu().numpy()

        # convert to list of vectors (pybind expects vector<Vector3d>)
        centers_list = [c for c in centers_np]
        normals_list = [n for n in normals_np]

        # Monte Carlo seed (important!)
        seed = np.random.randint(0, 1_000_000)

        exp_val = simulator.launch_simulation(
            int(seed),
            K,
            I,
            R,
            centers_list,
            normals_list
        )

        results.append(exp_val)

    return torch.tensor(results, dtype=torch.double).unsqueeze(-1)


# -------------------------
# INITIAL DATA
# -------------------------
def generate_initial_data(n=8):
    X = torch.rand(n, DIM, dtype=torch.double)
    X = BOUNDS[0] + (BOUNDS[1] - BOUNDS[0]) * X

    Y = objective(X)

    return X, Y


# -------------------------
# BO LOOP
# -------------------------
def run_bo(n_iter=20):

    X, Y = generate_initial_data()

    for i in range(n_iter):
        print(f"Iteration {i}")

        # Fit GP
        model = SingleTaskGP(X, Y)
        mll = ExactMarginalLogLikelihood(model.likelihood, model)
        fit_gpytorch_model(mll)

        # Acquisition function
        best_f = Y.min()   # minimizing dose
        EI = ExpectedImprovement(model, best_f=best_f)

        # Optimize acquisition
        candidate, _ = optimize_acqf(
            EI,
            bounds=BOUNDS,
            q=1,
            num_restarts=10,
            raw_samples=50,
        )

        # Evaluate
        new_Y = objective(candidate)

        # Update dataset
        X = torch.cat([X, candidate])
        Y = torch.cat([Y, new_Y])

        print("Best value so far:", Y.min().item())

    return X, Y


# -------------------------
# RUN
# -------------------------
if __name__ == "__main__":
    X, Y = run_bo(15)

    best_idx = torch.argmin(Y)
    best_x = X[best_idx]

    print("\nBest configuration:")
    print(best_x)
    print("Best dose:", Y[best_idx].item())