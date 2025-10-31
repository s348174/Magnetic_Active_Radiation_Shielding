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
        if (pow(R - sqrt(pow(X(0),2) + pow(X(1), 2)), 2) + pow(X(2),2) < pow(rho, 2)) {
            return true;
        }
        return false;
    }

    Vector3d torusMagneticField(const Vector3d& X) // This mehod computes the magnetic field in a given point X
    {
        Vector3d B;
        const double mu0 = 4 * M_PI * 1e-7; // Define magnetic constant
        const int N = 500;
        const double dtheta = M_PI / N; // Integration step

        if (isPointInTorus(X))
            return Vector3d::Zero();

        // Define array of points for integration
        const int numPoints = 2 * N + 1;
        VectorXd thetaVec(numPoints);
        for (int i = 0; i < numPoints; ++i)
            thetaVec(i) = i*dtheta;

        // Compute element-wise sin, cos, denom
        ArrayXd sinT = thetaVec.array().sin();
        ArrayXd cosT = thetaVec.array().cos();

        ArrayXd denom = (X(0) * X(0) + X(1) * X(1) + X(2) * X(2) + R * R
                         - 2 * R * X(0) * cosT
                         - 2 * R * X(1) * sinT).pow(1.5);  // element-wise ^(3/2)

        // Integrand funtions
        ArrayXd fx = X(2) * cosT / denom;
        ArrayXd fy = X(2) * sinT / denom;
        ArrayXd fz = (R - X(0) *  cosT - X(1) * sinT) / denom;

        // Simpson’s rule weights
        ArrayXd weights(numPoints);
        weights(0) = 1.0;
        weights(numPoints - 1) = 1.0;
        for (int i = 1; i < numPoints - 1; ++i)
            // Assign 2 if i is even, else 4
            if (i % 2 == 0) {weights(i) = 2;} else {weights(i) = 4;}

        // Simpson integration (vectorized)
        double integral_x = (weights * fx).sum() * dtheta / 3.0;
        double integral_y = (weights * fy).sum() * dtheta / 3.0;
        double integral_z = (weights * fz).sum() * dtheta / 3.0;

        // Magnetic field vector
        double coeff = mu0 * I * R / (4 * M_PI);
        B << coeff * integral_x, coeff * integral_y, coeff * integral_z;
        return B;
    }
};
