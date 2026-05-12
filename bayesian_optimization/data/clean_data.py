import numpy as np

INPUT_FILE = "hi_Flux_Point_1.dat"
OUTPUT_FILE = "data_clean.csv"
THRESHOLD = 1e-25  # tolerance for "zero"


def read_dat(filename):
    with open(filename, "r") as f:
        lines = f.readlines()

    # Remove comments
    data_lines = [l.strip() for l in lines if l.strip() and not l.startswith("C")]

    idx = 0

    # --- Energy ---
    start, n_energy = map(int, data_lines[idx].split())
    idx += 1

    energy = []
    while len(energy) < n_energy:
        energy.extend(map(float, data_lines[idx].split()))
        idx += 1

    energy = np.array(energy)

    # --- Particles ---
    start, n_particles = map(int, data_lines[idx].split())
    idx += 1

    particles = []
    while len(particles) < n_particles:
        particles.extend(data_lines[idx].split())
        idx += 1

    # --- Flux dimensions ---
    e1, e2, p1, p2 = map(int, data_lines[idx].split())
    idx += 1

    flux_values = []
    while len(flux_values) < n_energy * n_particles:
        flux_values.extend(map(float, data_lines[idx].split()))
        idx += 1

    flux = np.array(flux_values).reshape((n_particles, n_energy)).T

    return energy, particles, flux


def filter_particles(energy, particles, flux):
    mask = np.any(flux > THRESHOLD, axis=0)

    filtered_particles = [p for p, keep in zip(particles, mask) if keep]
    filtered_flux = flux[:, mask]

    return energy, filtered_particles, filtered_flux


def write_csv(filename, energy, particles, flux):
    with open(filename, "w") as f:
        # header
        f.write("Energy," + ",".join(particles) + "\n")

        # rows
        for i in range(len(energy)):
            row = [f"{energy[i]:.6e}"] + [f"{flux[i,j]:.6e}" for j in range(len(particles))]
            f.write(",".join(row) + "\n")


def main():
    energy, particles, flux = read_dat(INPUT_FILE)
    energy, particles, flux = filter_particles(energy, particles, flux)

    print(f"Original particles: {len(flux[0])}")
    print(f"Filtered particles: {len(particles)}")

    write_csv(OUTPUT_FILE, energy, particles, flux)
    print(f"Clean file written to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()