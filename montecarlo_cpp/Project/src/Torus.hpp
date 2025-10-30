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

    Vector3d torusMagneticField(const Vector3d& X) // This mehod computes the magnetic field in a given point X
    {
        Vector3d B;
        const double mu0 = 4 * M_PI * 1e-7; // Define magnetic constant
        const double dtheta = M_PI / N; // Integration step

        // Define array of points for integration
        const int N = 500;
        const int numPoints = 2 * N + 1; // Number of points for integration
        array<double, numPoints> theta;
        double step = 2 * M_PI / numPoints;
        size_t i = 0;
        while (i < numPoints) {
            theta[i] = i*step;
            i++;
        }

        // Convert array to Eigen vector
        VectorXd thetaVec(numPoints);
        for (int i = 0; i < numPoints; ++i) thetaVec(i) = theta[i];

        // Compute element-wise sin, cos, denom
        ArrayXd sinT = thetaVec.array().sin();
        ArrayXd cosT = thetaVec.array().cos();

        ArrayXd denom = (X(1) * X(1) + X(2) * X(2) + X(3) * X(3) + R * R
                         - 2 * R * X(1) * sinT
                         - 2 * R * X(2) * cosT).pow(1.5);  // element-wise ^(3/2)

        // Integrand funtions
        ArrayXd fx = X(3) * cosT / denom;
        ArrayXd fy = X(3) * sinT / denom;
        ArrayXd fz = (R - X(1)*  cosT - X(2) * sinT) / denom;

        // Simpson’s rule weights
        ArrayXd weights(numPoints);
        weights(0) = 1.0;
        weights(numPoints - 1) = 1.0;
        for (int i = 1; i < numPoints - 1; ++i)
            // Assign 2 if i is even, else 4
            if (i % 2 == 0) {weights(i) = 2;} else {weights(i) = 4;}

        // Simpson integration (vectorized)
        double integral_x = (weights * fx.matrix()).sum() * dtheta / 3.0;
        double integral_y = (weights * fy.matrix()).sum() * dtheta / 3.0;
        double integral_z = (weights * fz.matrix()).sum() * dtheta / 3.0;

        // Magnetic field vector
        double coeff = mu0 * I * R / (4 * M_PI);
        B << coeff * integral_x, coeff * integral_y, coeff * integral_z;
        return B;
    }
};
