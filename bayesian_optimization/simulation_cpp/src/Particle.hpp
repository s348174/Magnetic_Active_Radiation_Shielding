#pragma once

#include <Constants.hpp>
#include <Revelator.hpp>
#include <Eigen/Eigen>
#include <vector>
#include <Trajectory.hpp>
#include <BField.hpp>
#include <algorithm>
#include <math.h>
#include <cmath>

using namespace std;
using namespace Eigen;

struct Particle {
    double m; // Mass
    double q; // Charge
    Vector3d X_t; // Instant position
    Vector3d v_t; // Instant speed
    Vector3d a_t; // Instant acceleration
    Vector3d p_t; // Instant relativistic momentum
    double T_max; // Max time for simulation
    double dt; // Time step
    Trajectory tj; // Trajectory
    double hit_prob = 0; // Hit probability

    Particle(double m_val, double q_val, Vector3d X0, Vector3d v0, double T_val, double dt_val) { // Class constructor
        m = m_val;
        q = q_val;
        T_max = T_val;
        dt = dt_val;

        int N = round(T_max/dt);
        tj.X.reserve(N);
        tj.v.reserve(N);
        tj.a.reserve(N);

        X_t = X0;
        tj.X.push_back(X0);
        v_t = v0;
        tj.v.push_back(v0);
        a_t << 0, 0, 0;
        tj.a.push_back(a_t);
        // Init relativistic momentum p = gamma*m*v
        double v2 = v_t.squaredNorm();
        double gamma = 1.0 / sqrt(1.0 - min(v2 / (c_light*c_light), 0.999999999999)); // Avoid v >= c
        p_t = gamma * m * v_t;
        tj.p.push_back(p_t);
    }

    ~Particle() {} // Class destructor

    void updatePosition(BField& field, Revelator& revelator){ // Update the trajectory. Returns TRUE if the torus gets hit
        // Compute B field and Lorentz force
        Vector3d B = field.totalBField(X_t);

        // Adaptive step control
        const double dx_max = revelator.R; // Max displacement per step (m)
        const double dt_min = 1e-10;    // Min step size
        const double dt_max = 1e-4;     // Max step size
        double Bmag = B.norm();
        double vmag = v_t.norm();
        // Limit by displacement
        double dt_disp = dx_max / max(vmag, 1e-9);
        // Limit by 10% of cyclotron (gyration) period (if Bmag > tol): if B is small, we use bigger dt
        double dt_cycl = (Bmag > 1e-12) ? 0.1 * (2 * M_PI * m) / (abs(q) * Bmag) : dt_max;
        // Take the smaller of the two
        double dt_new = min({dt_disp, dt_cycl, dt_max});
        dt = clamp(dt_new, dt_min, dt_max); // Clamp between max and min to avoid too small or too big dt

        // Boris integrator
        Vector3d v_minus = v_t;

        // Rotation due to B
        Vector3d t = (q * B / m) * (0.5 * dt);
        Vector3d v_prime = v_minus + v_minus.cross(t);
        Vector3d v_plus = v_minus + (2.0 / (1.0 + t.squaredNorm())) * (v_prime.cross(t));

        // Half acceleration from E
        v_t = v_plus;

        // Update position
        X_t += v_t * dt;

        // Update
        a_t = q * v_t.cross(B) / m;
        tj.p.push_back(p_t);
        tj.a.push_back(a_t);
        tj.v.push_back(v_t);
        tj.X.push_back(X_t);

        // Compute probability to be detected
        if (X.norm() < revelator.R) hit_prob += revelator.revelatorProbability(X_t) * dt;
    }
};
