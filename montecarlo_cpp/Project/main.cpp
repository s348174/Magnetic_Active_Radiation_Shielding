#include "Utils.hpp"
#include <Eigen/Eigen>
#include <iostream>
#include <Torus.hpp>
#include <Particle.hpp>
#include <Utils.hpp>

using namespace std;
using namespace Eigen;

int main()
{
    // Simulation arguments
    double N = 1000;
    const double T = 1e7; // K
    double dt = 1e-9;
    unsigned long seed = 1002;

    // Define Torus
    double R = 20;
    double rho = 1.5;
    double I = 1e4;
    Torus torus;
    torus.R = R;
    torus.rho = rho;
    torus.I = I;

    runFromCSV_MT("../Project/particles_input.csv", torus, N, T, dt, seed);

    // if (monteCarlo(torus, m, q, N, T, dt))
    //     cout << "Simulation completed successfully!" << endl;
    return 0;
}
