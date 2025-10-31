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
    double N = 100;
    vector<double> v_samples = sampleMbSpeed(m, N);
    double q = 1.6e-19;
    double I = 1e5;
    double dt = 1e-7;
    double T_max = 1e-3;
    Vector3d v0(0,-1e5,0);
    Vector3d X0(0,100,0);
    double R = 20;
    double rho = 1.5;
    Torus torus;
    torus.R = R;
    torus.rho = rho;
    torus.I = I;

    Particle part(m, q, X0, v0, T_max, dt);
    double t = 0;
    while (!part.updatePosition(torus) && t < T_max) {
        t += dt;
    }
    if (!part.hit) {
        cout << "Final time: " << t << endl;
        cout << "Particle deviated!" << endl;
        cout << "Final position: " << part.X_t(0) << ", " << part.X_t(1) << ", " << part.X_t(2) << endl;
    }
    else {
        cout << "Particle hit the torus!" << endl;
    }
    return 0;
}
