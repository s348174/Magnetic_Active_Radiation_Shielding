#include "Utils.hpp"
#include <Eigen/Eigen>
#include <iostream>
#include <Torus.hpp>
#include <Particle.hpp>
#include <Utils.hpp>
#include <chrono>

using namespace std;
using namespace Eigen;

int main()
{
    chrono::steady_clock::time_point t_begin = chrono::steady_clock::now();
    // Simulation arguments
    double N = 1e3; // Number of simulated particles
    const double T = 1e7; // K
    double dt = 1e-9; // Initial time step
    unsigned long seed = 67;

    // Define Torus
    double R = 10;
    double rho = 1;
    double I = 1e6;
    Torus torus;
    torus.R = R;
    torus.rho = rho;
    torus.I = I;

    // Run multithread simulation from CSV input (for particles phyisics data)
    runFromCSV_MT("particles_input.csv", torus, N, T, dt, seed);

    chrono::steady_clock::time_point t_end = chrono::steady_clock::now();
    double elapsedTime = chrono::duration_cast<chrono::milliseconds>(t_end-t_begin).count();
    cout << "Elapsed simulation time: " << elapsedTime << "ms." << endl;
    return 0;
}
