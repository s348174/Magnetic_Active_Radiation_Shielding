#include "Utils.hpp"
#include <Eigen/Eigen>
#include <iostream>
#include <Torus.hpp>
#include <Particle.hpp>

using namespace std;
using namespace Eigen;

int main()
{
    double m = 1.67e-27;
    // double m = 9.1e-31;
    double q = 1.6e-19;
    double N = 100;
    double dt = 1e-7;
    // double T_max = 1e-3;
    // Vector3d v0(0,-1e5,0);
    // Vector3d X0(0,100,0);
    double R = 20;
    double rho = 1.5;
    double I = 1e4;
    Torus torus;
    torus.R = R;
    torus.rho = rho;
    torus.I = I;

    if (monteCarlo(torus, m, q, N, dt))
        cout << "Simulation completed successfully!" << endl;
    return 0;
}
