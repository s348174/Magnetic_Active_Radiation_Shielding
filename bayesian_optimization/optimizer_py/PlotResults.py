import matplotlib

matplotlib.use("QtAgg")

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import pandas as pd


def coil_points(center, normal, radius=0.1, num_points=100):
    center = np.asarray(center, dtype=float)
    normal = np.asarray(normal, dtype=float)

    normal_norm = np.linalg.norm(normal)
    if normal_norm == 0:
        raise ValueError("Coil normal vector must be non-zero")
    normal = normal / normal_norm

    reference = np.array([0.0, 0.0, 1.0])
    if abs(np.dot(normal, reference)) > 0.9:
        reference = np.array([0.0, 1.0, 0.0])

    basis_u = np.cross(normal, reference)
    basis_u_norm = np.linalg.norm(basis_u)
    if basis_u_norm == 0:
        raise ValueError("Failed to build an in-plane basis for the coil")
    basis_u = basis_u / basis_u_norm
    basis_v = np.cross(normal, basis_u)

    theta = np.linspace(0, 2 * np.pi, num_points)
    circle = (
        center[:, None]
        + radius * np.cos(theta)[None, :] * basis_u[:, None]
        + radius * np.sin(theta)[None, :] * basis_v[:, None]
    )
    return circle[0], circle[1], circle[2]

def interactive_plot(result_file="configurations/best_configuration_K10_42_converged.csv"):
    df = pd.read_csv(result_file)

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")

    # Plot a sphere of radius 1.
    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, np.pi, 100)
    x = np.outer(np.cos(u), np.sin(v))
    y = np.outer(np.sin(u), np.sin(v))
    z = np.outer(np.ones(np.size(u)), np.cos(v))
    ax.plot_surface(x, y, z, color="lightblue", alpha=0.5)

    for i in range(df.shape[0]):
        # Plot the coil centers as red points.
        x = df["cX"][i]
        y = df["cY"][i]
        z = df["cZ"][i]
        ax.scatter(x, y, z, color="red", label=f"Configuration {i + 1}")

        #Plot the coils normal vectors as arrows.
        nX = df["nX"][i]
        nY = df["nY"][i]
        nZ = df["nZ"][i]
        ax.quiver(x, y, z, nX, nY, nZ, length=0.2, color="orange")

        # Compute the coil points as a circle on the normal plane to the normal vector.
        coil_x, coil_y, coil_z = coil_points(
            center=(x, y, z),
            normal=(nX, nY, nZ),
            radius=0.5,
            num_points=100,
        )
        ax.plot(coil_x, coil_y, coil_z, color="green")
        
        
    def update(frame):
        ax.view_init(elev=20, azim=frame)
        return ax,

    rotation = FuncAnimation(fig, update, frames=np.arange(0, 360, 2), interval=50, blit=False)
    plt.show()
    return fig, ax, rotation

def plot_convergence_metrics(ei_array, best_mean_hist, variance_hist):
    plt.figure(figsize=(10, 6))
    plt.plot(ei_array, marker='o')
    plt.title("Acquisition function over time")
    plt.xlabel("Iteration")
    plt.ylabel("Acquisition Function Value")
    plt.show()

    plt.figure(figsize=(10, 6))
    plt.plot(best_mean_hist, marker='o')
    plt.title("Best Predicted Mean over time")
    plt.xlabel("Iteration")
    plt.ylabel("Best Predicted Mean")
    plt.show()

    plt.figure(figsize=(10, 6))
    plt.plot(variance_hist, marker='o')
    plt.title("Predicted Variance over time")
    plt.xlabel("Iteration")
    plt.ylabel("Predicted Variance")
    plt.show()

if __name__ == "__main__":
    interactive_plot()
