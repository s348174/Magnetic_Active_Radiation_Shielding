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
    double N = 1e4;
    const double T = 1e7; // K
    double dt = 1e-9;
    unsigned long seed = 10000;

    // Define Torus
    double R = 20;
    double rho = 1.5;
    double I = 1e4;
    Torus torus;
    torus.R = R;
    torus.rho = rho;
    torus.I = I;

    // Run multithread simulation from CSV input
    runFromCSV_MT("../Project/particles_input.csv", torus, N, T, dt, seed);

    chrono::steady_clock::time_point t_end = chrono::steady_clock::now();
    double elapsedTime = chrono::duration_cast<chrono::milliseconds>(t_end-t_begin).count();
    cout << "Elapsed simulation time: " << elapsedTime << "ms." << endl;
    return 0;
}
