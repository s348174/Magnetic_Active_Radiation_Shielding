# pragma once

#include <Constants.hpp>
#include <Eigen/Eigen>
#include <math.h>
#include <cmath>
#include <algorithm>
#include <vector>
#include <stdexcept>

using namespace std;
using namespace Eigen;

struct BField {
    size_t k; // Number of coils
    vector<Vector3d> centers; // Coils centers
    vector<Vector3d> normals; // Coils normals
    double R; // Coils radius
    double I; // Coils current intensity
    double coeff; // Integration coefficient
    vector<Quaterniond> rotations; // Store quaternions for rotations of normals w.r.t. z
    vector<Quaterniond> conjugates; // Store rotation conjugates

    // Static cached data for Biot-Savart (initialized once)
    static constexpr int N = 300;
    static constexpr int numPoints = 2 * N + 1;
    static constexpr double dtheta = PI / N; // Integration step
    static inline ArrayXd sinT, cosT, weights;
    static inline bool precomputedBS = false;

    // Static cached data for SAM (initialized once)
    static inline double C;
    static inline bool precomputedSAM = false;

    static void precomputeTablesBS() {
        if (precomputedBS) return;

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
        precomputedBS = true;
    }

    static void precomputeTablesSAM() {
        if(precomputedSAM) return;
        C = mu0 / PI;
        precomputedSAM = true;
    }

    BField(size_t k_in, vector<Vector3d> centers_in, vector<Vector3d> normals_in, double R_in, double I_in) {
        if(k_in != centers_in.size() || k_in != normals_in.size()) {
            throw invalid_argument("Error in generating BField.");
            return;
        }

        k = k_in;
        centers.reserve(k);
        normals.reserve(k);
        for (auto& v : centers_in) {
            centers.push_back(v);
        }
        for (auto& v : normals_in) {
            normals.push_back(v);
        }

        R = R_in;
        I = I_in;
        coeff = mu0 * I * R / (4 * PI);

        // Precompute rotation of frames of reference
        Vector3d z(0, 0, 1);
        for (size_t i = 0; i < k; ++i) {
            Quaterniond q = Quaterniond::FromTwoVectors(z, normals[i].normalized());
            rotations.push_back(q);
            conjugates.push_back(q.conjugate());
        }
    }

    Vector3d computeBS(const Vector3d& X) {
        // This mehod computes the magnetic field of a single coil in a given point X with Biot-Savart law
        // Precompute static variables
        precomputeTablesBS();
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

    Vector3d computeSAM(const Vector3d& X) {
        // Compute magnetic field with SAM algorithm
        precomputeTablesSAM();

        // Conversion to cylindrical coordinates
        const double x = X(0);
        const double y = X(1);
        const double rho2 = x * x + y * y;
        const double rho = sqrt(rho2);
        const double z = X(2);
        const double z2 = z * z;
        const double R2 = R * R;

        // Axisymmetric closed form on the coil axis avoids 0/0 in B_rho.
        if (rho < 1e-9) {
            const double denom_axis = pow(R2 + z2, 1.5);
            Vector3d B;
            B << 0.0, 0.0, mu0 * I * R2 / (2.0 * denom_axis);
            return B;
        }

        // Constants for integration
        const double a2 = R2 + rho2 + z2 - 2*R*rho;
        const double b2 = R2 + rho2 + z2 + 2*R*rho;
        const double b = sqrt(b2);
        const double k2 = std::max(0.0, std::min(1.0, 1.0 - (a2 / b2)));
        const double k = sqrt(k2);

        // Compute complete elliptic integrals (K: first kind, E: second kind)
        const double K = comp_ellint_1(k);
        const double E = comp_ellint_2(k);

        // Compute field components
        const double term1_rho = (R2 + rho2 + z2);
        const double term1_z = (R2 - rho2 - z2);
        const double denominator = 2.0 * a2 * b;

        double B_rho = C * I * z * (term1_rho * E - a2 * K) / (denominator * rho);
        double B_z = C * I * (term1_z * E + a2 * K) / denominator;
        // B_phi is 0

        // Transform back in cartesian
        Vector3d B;
        B << B_rho * (x / rho), B_rho * (y / rho), B_z;

        return B;
    }

    Vector3d totalBField(const Vector3d& X) {
        // This methods just sums up the field over all coils
        Vector3d totalB(0.0, 0.0, 0.0);
        for (size_t i = 0; i < k; ++i) {
            Vector3d X_local = conjugates[i] * (X - centers[i]); // Rotate and recenter
            // totalB += rotations[i] * coilBField(X_local); // Compute field in frame of reference and add to total field
            totalB += rotations[i] * computeSAM(X_local); // Compute field with SAM algorithm and rotate
        }
        return totalB;
    }
};
