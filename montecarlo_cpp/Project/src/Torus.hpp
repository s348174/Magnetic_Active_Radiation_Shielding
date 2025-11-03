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
    double I; // Current

    // Static cached data (initialized once)
    static constexpr int N = 500;
    static constexpr int numPoints = 2 * N + 1;
    static constexpr double mu0 = 4 * M_PI * 1e-7; // Define magnetic constant
    static constexpr double dtheta = M_PI / N; // Integration step
    static inline ArrayXd sinT, cosT, weights;
    static inline bool precomputed = false;

    static void precomputeTables() {
        if (precomputed) return;
        VectorXd thetaVec(numPoints);
        for (int i = 0; i < numPoints; ++i)
            thetaVec(i) = i * dtheta;

        sinT = thetaVec.array().sin();
        cosT = thetaVec.array().cos();

        weights.resize(numPoints);
        weights(0) = 1.0;
        weights(numPoints - 1) = 1.0;
        for (int i = 1; i < numPoints - 1; ++i)
            // Assign 2 if i is even, else 4
            if (i % 2 == 0) weights(i) = 2; else weights(i) = 4;

        precomputed = true;
    }

    bool isPointInTorus(const Vector3d& X) // This method checks if a given point X is inside the torus
    {
        if (pow(R - sqrt(pow(X(0),2) + pow(X(1), 2)), 2) + pow(X(2),2) < pow(rho, 2)) {
            return true;
        }
        return false;
    }

    Vector3d torusMagneticField(const Vector3d& X) // This mehod computes the magnetic field in a given point X
    {
        // Return zero field if point is inside torus
        if (isPointInTorus(X))
            return Vector3d::Zero();

        // Precompute static variables
        precomputeTables();
        Vector3d B;

        ArrayXd base = (X(0) * X(0) + X(1) * X(1) + X(2) * X(2) + R * R
                         - 2 * R * X(0) * cosT
                         - 2 * R * X(1) * sinT);
        ArrayXd denom = base * base.sqrt(); // Equivalent to pow(1.5), but faster

        // Integrand funtions
        ArrayXd fx = X(2) * cosT / denom;
        ArrayXd fy = X(2) * sinT / denom;
        ArrayXd fz = (R - X(0) *  cosT - X(1) * sinT) / denom;

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
