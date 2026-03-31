# pragma once

#include <Constants.hpp>
#include <Eigen/Eigen>
#include <math.h>
#include <cmath>
#include <vector>
//#include <stdexcept>

using namespace std;
using namespace Eigen;

struct BField {
    size_t k; // Number of coils
    vector<Vector3d> centers; // Coils centers
    vector<Vector3d> normals; // Coils normals
    double R; // Coils radius
    double I; // Coils current intensity
    double coeff; // Integration coefficient
    vector<Quaterniond> rotations; // Store rotations of normals w.r.t. z

    // Static cached data (initialized once)
    static constexpr int N = 300;
    static constexpr int numPoints = 2 * N + 1;
    static constexpr double dtheta = PI / N; // Integration step
    static inline ArrayXd sinT, cosT, weights;
    static inline bool precomputed = false;

    static void precomputeTables() {
        if (precomputed) return;

        // Precompute arrays for integration
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

    BField(double k_in, vector<Vector3d> centers_in, vector<Vector3d> normals_in, double R_in, double I_in) {
        k = k_in;
        centers = centers_in;
        normals = normals_in;
        // if(k != centers.size() || k != normals.size())
        //     throw invalid_argument("Error in generating BField.");
        //     return;
        R = R_in;
        I = I_in;
        coeff = mu0 * I * R / (4 * PI);

        // Rotate frame of reference
        Vector3d z(0, 0, 1);
        for (size_t i = 0; i < k; ++i)
            rotations.push_back(Quaterniond::FromTwoVectors(z, normals[i].normalized()));
    }

    Vector3d coilBField(const Vector3d& X) {
        // This mehod computes the magnetic field of a single coil in a given point X
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
        B << coeff * integral_x, coeff * integral_y, coeff * integral_z;
        return B;
    }

    Vector3d totalBField(const Vector3d& X) {
        // This methods just sums up the field over all coils
        Vector3d totalB;
        for (size_t i = 0; i < k; ++i) {
            Vector3d X_local = rotations[i].conjugate() * (X - centers[i]); // Rotate and recenter
            totalB += rotations[i] * coilBField(X_local); // Compute field in frame of reference and add to total field
        }
        return totalB;
    }
};
