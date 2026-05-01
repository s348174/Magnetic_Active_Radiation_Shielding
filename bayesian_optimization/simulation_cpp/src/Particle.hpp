#pragma once

#include <Constants.hpp>
#include <Revelator.hpp>
#include <Eigen/Eigen>
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
    // Vector3d a_t; // Instant acceleration
    // Vector3d p_t; // Instant relativistic momentum
    double T_max; // Max time for simulation
    double dt; // Time step
    // Trajectory tj; // Trajectory
    double hit_prob; // Hit probability
    const double dt_min = 1e-10;    // Min step size
    const double dt_max = 1e-4;     // Max step size

    Particle(double m_val, double q_val, Vector3d X0, Vector3d v0, double T_val, double dt_val) { // Class constructor
        m = m_val;
        q = q_val;
        T_max = T_val;
        dt = dt_val;
        X_t = X0;
        v_t = v0;

        // Set hit prob to 0
        hit_prob = 0.0;
    }

    ~Particle() {} // Class destructor

    void updatePosition(BField& field, Revelator& revelator){ // Update the trajectory. Returns TRUE if the torus gets hit
        // Compute B field and Lorentz force
        Vector3d B = field.totalBField(X_t);

        // Adaptive step control
        const double dx_max = revelator.R / 4; // Max displacement per step (m)
        double Bmag = B.norm();
        double vmag = v_t.norm();
        // Limit by displacement
        double dt_disp = dx_max / max(vmag, 1e-9);
        // Limit by 10% of cyclotron (gyration) period (if Bmag > tol): if B is small, we use bigger dt
        double dt_cycl = (Bmag > 1e-12) ? 0.1 * (2 * M_PI * m) / (abs(q) * Bmag) : dt_max;
        // Take the smaller of the two
        double dt_new = min({dt_disp, dt_cycl, dt_max});
        dt = clamp(dt_new, dt_min, dt_max); // Clamp between max and min to avoid too small or too big dt

        // Boris integrator simplified (no E field)
        Vector3d t = (q * B / m) * (0.5 * dt);
        Vector3d s = 2.0 * t / (1.0 + t.squaredNorm());
        Vector3d v_minus = v_t;
        Vector3d v_prime = v_minus + v_minus.cross(t);
        Vector3d v_plus = v_minus + v_prime.cross(s);
        Vector3d v_next = v_plus;

        // Compute probability to be detected at this time step
        Vector3d X_next = X_t + dt * v_next; // Next position
        // Quadrature along the path
        hit_prob += dt * (revelator.revelatorProbability(X_t) +
                          revelator.revelatorProbability(X_next)) / 2;

        // Update position and speed
        X_t = X_next;
        v_t = v_next;
    }

    void updatePositionRK4(BField& field, Revelator& revelator){ 
        // Update the trajectory with RK4. Returns TRUE if the torus gets hit
        // Compute B field and Lorentz force
        Vector3d B = field.totalBField(X_t);

        // Adaptive step control
        const double dx_max = revelator.R / 4; // Max displacement per step (m)
        double Bmag = B.norm();
        double vmag = v_t.norm();
        // Limit by displacement
        double dt_disp = dx_max / max(vmag, 1e-9);
        // Limit by 10% of cyclotron (gyration) period (if Bmag > tol): if B is small, we use bigger dt
        double dt_cycl = (Bmag > 1e-12) ? 0.1 * (2 * M_PI * m) / (abs(q) * Bmag) : dt_max;
        // Take the smaller of the two
        double dt_new = min({dt_disp, dt_cycl, dt_max});
        dt = clamp(dt_new, dt_min, dt_max); // Clamp between max and min to avoid too small or too big dt

        // Compute acceleration at current position
        Vector3d a_t = (q / m) * v_t.cross(B);

        // RK4 integration for position and velocity
        Vector3d k1_v = a_t;
        Vector3d k1_x = v_t;

        Vector3d k2_v = (q / m) * (v_t + 0.5 * dt * k1_v).cross(field.totalBField(X_t + 0.5 * dt * k1_x));
        Vector3d k2_x = v_t + 0.5 * dt * k1_v;

        Vector3d k3_v = (q / m) * (v_t + 0.5 * dt * k2_v).cross(field.totalBField(X_t + 0.5 * dt * k2_x));
        Vector3d k3_x = v_t + 0.5 * dt * k2_v;

        Vector3d k4_v = (q / m) * (v_t + dt * k3_v).cross(field.totalBField(X_t + dt * k3_x));
        Vector3d k4_x = v_t + dt * k3_v;

        Vector3d v_next = v_t + (dt / 6.0) * (k1_v + 2.0 * k2_v + 2.0 * k3_v + k4_v);
        Vector3d X_next = X_t + (dt / 6.0) * (k1_x + 2.0 * k2_x + 2.0 * k3_x + k4_x);

        // Quadrature along the path
        hit_prob += dt * (revelator.revelatorProbability(X_t) +
                          revelator.revelatorProbability(X_next)) / 2;

        // Update position and speed
        X_t = X_next;
        v_t = v_next;
    }

    void updateBS(BField& field, Revelator& revelator){ // Update the trajectory. Returns TRUE if the torus gets hit
        // Compute B field and Lorentz force
        Vector3d B = field.totalBS(X_t);

        // Adaptive step control
        const double dx_max = revelator.R / 4; // Max displacement per step (m)
        double Bmag = B.norm();
        double vmag = v_t.norm();
        // Limit by displacement
        double dt_disp = dx_max / max(vmag, 1e-9);
        // Limit by 10% of cyclotron (gyration) period (if Bmag > tol): if B is small, we use bigger dt
        double dt_cycl = (Bmag > 1e-12) ? 0.1 * (2 * M_PI * m) / (abs(q) * Bmag) : dt_max;
        // Take the smaller of the two
        double dt_new = min({dt_disp, dt_cycl, dt_max});
        dt = clamp(dt_new, dt_min, dt_max); // Clamp between max and min to avoid too small or too big dt

        // Boris integrator simplified (no E field)
        Vector3d t = (q * B / m) * (0.5 * dt);
        Vector3d s = 2.0 * t / (1.0 + t.squaredNorm());
        Vector3d v_minus = v_t;
        Vector3d v_prime = v_minus + v_minus.cross(t);
        Vector3d v_plus = v_minus + v_prime.cross(s);
        Vector3d v_next = v_plus;

        // Compute probability to be detected at this time step
        Vector3d X_next = X_t + dt * v_next; // Next position
        // Quadrature along the path
        hit_prob += dt * (revelator.revelatorProbability(X_t) +
                          revelator.revelatorProbability(X_next)) / 2;

        // Update position and speed
        X_t = X_next;
        v_t = v_next;
    }

    void updatePositionRK4_BS(BField& field, Revelator& revelator){
        // Update the trajectory with RK4. Returns TRUE if the torus gets hit
        // Compute B field and Lorentz force
        Vector3d B = field.totalBS(X_t);

        // Compute acceleration at current position
        Vector3d a_t = (q / m) * v_t.cross(B);

        // RK4 integration for position and velocity
        Vector3d k1_v = a_t;
        Vector3d k1_x = v_t;

        Vector3d k2_v = (q / m) * (v_t + 0.5 * dt * k1_v).cross(field.totalBField(X_t + 0.5 * dt * k1_x));
        Vector3d k2_x = v_t + 0.5 * dt * k1_v;

        Vector3d k3_v = (q / m) * (v_t + 0.5 * dt * k2_v).cross(field.totalBField(X_t + 0.5 * dt * k2_x));
        Vector3d k3_x = v_t + 0.5 * dt * k2_v;

        Vector3d k4_v = (q / m) * (v_t + dt * k3_v).cross(field.totalBField(X_t + dt * k3_x));
        Vector3d k4_x = v_t + dt * k3_v;

        Vector3d v_next = v_t + (dt / 6.0) * (k1_v + 2.0 * k2_v + 2.0 * k3_v + k4_v);
        Vector3d X_next = X_t + (dt / 6.0) * (k1_x + 2.0 * k2_x + 2.0 * k3_x + k4_x);

        // Quadrature along the path
        hit_prob += dt * (revelator.revelatorProbability(X_t) +
                          revelator.revelatorProbability(X_next)) / 2;

        // Update position and speed
        X_t = X_next;
        v_t = v_next;
    }
};
