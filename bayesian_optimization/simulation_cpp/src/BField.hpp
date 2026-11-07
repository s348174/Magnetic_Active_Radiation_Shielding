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
    static constexpr double wireRadius = 1e-3; // Conductor core radius: 1 mm.
    size_t k; // Number of coils
    vector<Vector3d> centers; // Coils centers
    vector<Vector3d> normals; // Coils normals
    double R; // Coils radius
    double I; // Coils current intensity
    double coeff; // Integration coefficient
    vector<Quaterniond> rotations; // Store quaternions for rotations of normals w.r.t. z
    vector<Quaterniond> conjugates; // Store rotation conjugates

    // Static cached data for Biot-Savart (initialized once)
    static constexpr int N = 500;
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
        if (!std::isfinite(R_in) || R_in <= wireRadius)
            throw invalid_argument("Coil radius must exceed the 1 mm wire radius.");
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

    // Local straight-wire approximation with uniform current density inside the conductor.
    // A jump at the wire surface is allowed; the curved coil's background field is omitted here.
    bool computeWireField(const Vector3d& X, Vector3d& B) const {
        const double rho = std::hypot(X(0), X(1));
        const double dr = rho - R;
        if (dr * dr + X(2) * X(2) >= wireRadius * wireRadius) return false;
        // R > wireRadius guarantees rho > 0 inside this tube.
        const double factor = mu0 * I / (2.0 * PI * wireRadius * wireRadius);
        B << factor * X(2) * X(0) / rho,
             factor * X(2) * X(1) / rho,
            -factor * dr;
        return true;
    }

    Vector3d computeBS(const Vector3d& X) {
        // This mehod computes the magnetic field of a single coil in a given point X with Biot-Savart law
        Vector3d B;
        if (computeWireField(X, B)) return B;
        // Precompute static variables
        precomputeTablesBS();

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
        Vector3d B;
        if (computeWireField(X, B)) return B;
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
        // Retain numerical guards outside the finite conductor core.
        if (a2 < 1e-12 || b2 < 1e-12 || !std::isfinite(a2) || !std::isfinite(b2)) {
            return Vector3d::Zero();
        }
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
        if (denominator < 1e-12 || !std::isfinite(denominator)) {
            return Vector3d::Zero();
        }

        double B_rho = C * I * z * (term1_rho * E - a2 * K) / (denominator * rho);
        double B_z = C * I * (term1_z * E + a2 * K) / denominator;
        // B_phi is 0

        // Transform back in cartesian
        const double rho_safe = std::max(rho, 1e-12);
        B << B_rho * (x / rho_safe), B_rho * (y / rho_safe), B_z;

        return B;
    }

    Vector3d totalBField(const Vector3d& X) {
        // This methods just sums up the field over all coils
        Vector3d totalB(0.0, 0.0, 0.0);
        for (size_t i = 0; i < k; ++i) {
            Vector3d X_local = conjugates[i] * (X - centers[i]); // Rotate and recenter
            totalB += rotations[i] * computeSAM(X_local); // Compute field with SAM algorithm and rotate
        }
        return totalB;
    }
    Vector3d totalBS(const Vector3d& X) {
        // This methods just sums up the field over all coils
        Vector3d totalB(0.0, 0.0, 0.0);
        for (size_t i = 0; i < k; ++i) {
            Vector3d X_local = conjugates[i] * (X - centers[i]); // Rotate and recenter
            totalB += rotations[i] * computeBS(X_local); // Compute field with SAM algorithm and rotate
        }
        return totalB;
    }
};
