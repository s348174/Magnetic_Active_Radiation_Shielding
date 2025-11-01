#pragma once

#include <Eigen/Eigen>
#include <vector>
#include <Trajectory.hpp>
#include <Torus.hpp>
#include <iostream>

using namespace std;
using namespace Eigen;

struct Particle {
    double m; // Mass
    double q; // Charge
    Vector3d X_t; // Instant position
    Vector3d v_t; // Instant speed
    Vector3d a_t; // Instant acceleration
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

        hit = false;
    }

    ~Particle() {} // Class destructor

    bool isParticleInTorus(Torus& torus) { // Verifies if the particle hit the torus
        if (torus.isPointInTorus(X_t)) {
            return true;
        }
        // To reduce risk of missing due to overshooting, we check also for the mean next expected position
        Vector3d X_next = X_t + dt * v_t + dt * dt * a_t / 2;
        Vector3d X_mean = (X_t + X_next) / 2;
        return torus.isPointInTorus(X_mean);
    }

    bool updatePosition(Torus& torus){ // Update the trajectory. Returns TRUE if the torus gets hit
        Vector3d B = torus.torusMagneticField(X_t); // Compute the B field in X_t
        Vector3d F_L = q * v_t.cross(B); // Compute Lorenz force
        // Update acceleration, speed and position
        a_t = F_L / m;
        v_t = v_t + dt * a_t;
        X_t = X_t + dt * (v_t + tj.v.back()) / 2;

        // Update trajectory
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
