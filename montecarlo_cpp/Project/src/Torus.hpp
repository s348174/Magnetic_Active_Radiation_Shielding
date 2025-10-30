#pragma once

#include <vector>
#include <Eigen/Eigen>
#include <math.h>
#include <cmath>
#include <numbers>
#include <array>

using namespace std;
using namespace Eigen;

struct Torus {
    double R; // Main radius
    double rho; // Internal radius
    double I; // Current intensity

    bool isPointInTorus(const Vector3d& X) // This method checks if a given point X is inside the torus
    {
        if (pow(R - sqrt(pow(X(1),2) + pow(X(2), 2)), 2) + pow(X(3),2) < pow(rho, 2)) {
            return true;
        }
        return false;
    }

    Vector3d torusManeticField(const Vector3d& X) // This mehod computes the magnetic field in a given point X
    {
        Vector3d B;
        const double mu0 = 4 * pi * 1e-7; // Define magnetic constant

        // Define array of points for integration
        const int N = 500; // Number of points for integration
        array<double, 2*N + 1> theta;
        double a = 0;
        double step = 2*M_PI/(2*N+1);
        size_t i = 0;
        while (i < theta.size() && a <= 2*PI) {
            theta[i] = a;
            a += step;
            i++;
        }
        return B;
    }
};
