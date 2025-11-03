#pragma once

#include <Eigen/Eigen>
#include <vector>
#include <Trajectory.hpp>
#include <Torus.hpp>
#include <iostream>
#include <algorithm>
#include <math.h>
#include <cmath>

const double c_light = 299792458.0; // m/s

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
    bool hit;

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

        hit = false;
    }

    ~Particle() {} // Class destructor

    bool isParticleInTorus(Torus& torus) { // Verifies if the particle hit the torus
        if (torus.isPointInTorus(X_t)) {
            return true;
        }
        // To reduce risk of missing due to overshooting, we check also for the mean next expected position
        // Vector3d X_next = X_t + dt * v_t + dt * dt * a_t / 2;
        // Vector3d X_mean = (X_t + X_next) / 2;
        // return torus.isPointInTorus(X_mean);
        return false;
    }

    bool updatePosition(Torus& torus){ // Update the trajectory. Returns TRUE if the torus gets hit
        // Compute B field and Lorentz force
        Vector3d B = torus.torusMagneticField(X_t);
        // Vector3d B = Vector3d::Zero();
        Vector3d E = Vector3d::Zero(); // if you have E; default zero

        // Adaptive step control
        const double dx_max = 1e0;      // Max displacement per step (m)
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
        // Half acceleration from E
        Vector3d v_minus = v_t + (q * E / m) * (0.5 * dt);

        // Rotation due to B
        Vector3d t = (q * B / m) * (0.5 * dt);
        Vector3d v_prime = v_minus + v_minus.cross(t);
        Vector3d v_plus = v_minus + (2.0 / (1.0 + t.squaredNorm())) * (v_prime.cross(t));

        // Half acceleration from E
        v_t = v_plus + (q * E / m) * (0.5 * dt);

        // CONTROL: EULER METHOD
        // Vector3d F_L = q * (E + v_t.cross(B));
        // Vector3d F_L = Vector3d::Zero();
        // a_t = F_L / m;
        // v_t = v_t + (a_t + tj.a.back()) / 2;

        // Update position
        X_t += v_t * dt;

        // Update
        a_t = q * (E + v_t.cross(B)) / m;
        tj.p.push_back(p_t);
        tj.a.push_back(a_t);
        tj.v.push_back(v_t);
        tj.X.push_back(X_t);

        if (isParticleInTorus(torus)) {
            hit = true;
            return hit;
        }
        return hit;
    }
};
