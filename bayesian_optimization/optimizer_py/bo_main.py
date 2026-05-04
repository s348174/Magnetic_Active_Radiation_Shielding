# Import libraries for optimization
import torch

# Import standard libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import csv
import time

# Import externals functions
from PlotResults import interactive_plot, plot_convergence_metrics
from BoUtils import unpack_configuration, denormalize
from BoLoop import bo_matern_kernel, bo_rbf_kernel
from Objective import spherical_to_cartesian

from input import K, SEED, device

def main():
    print(f"Using device: {device}")
    start_clock = time.time()

    # Run BO optimization loop
    X_train, y_train, ei_array, best_mean_hist, variance_hist = bo_matern_kernel(nu=2.5)
    #X_train, y_train, ei_array, best_mean_hist, variance_hist = bo_rbf_kernel()
    
    end_clock = time.time() - start_clock
    print(f'Elapsed time: {end_clock:.4f} s')

    # Final results
    best_index = np.argmin(-y_train) # Get index of best energy (negate back)
    best_X_norm = X_train[best_index]
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
    
    # Plot convergence metrics
    plot_convergence_metrics(ei_array, best_mean_hist, variance_hist)

    # Plot the best configuration
    interactive_plot(result_file=f"configurations/best_configuration_K{K}_{SEED}.csv")

if __name__ == "__main__":    main()