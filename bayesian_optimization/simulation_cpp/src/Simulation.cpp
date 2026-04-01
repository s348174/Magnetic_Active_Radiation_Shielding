#include "Simulation.hpp"
#include "Utils.hpp"
#include <BField.hpp>
#include <Revelator.hpp>
#include <Eigen/Eigen>
#include <vector>

using namespace std;
using namespace Eigen;

// Minimal placeholder function
double launch_simulation(unsigned long seed, int K, double I, double R, vector<Vector3d> centers, vector<Vector3d> normals) {
    // Simulation specifics
    double N = 1e5; // Number of simulated particles
    const double T = 1e7; // K
    double dt = 1e-9; // Initial time step

    // Define Revelator
    double rho = 10;
    Revelator revelator(rho);

    // Define expected dose
    double totalExpectedDose;

    // Try to run simulation
    try {
        // Build magnetic field
        BField field(K, centers, normals, R, I);

        // Run multithread simulation from CSV input (for particles phyisics data)
        totalExpectedDose = runFromCSV_MT("../simulation_cpp/particles_input.csv", field, revelator, N, T, dt, seed);
    } catch (const invalid_argument e){
        cerr << "Error: number of coils and number of coils parameters provided do not match!" << endl;
    }

    return totalExpectedDose;
}
