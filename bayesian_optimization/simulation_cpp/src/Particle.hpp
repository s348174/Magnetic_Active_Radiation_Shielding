#pragma once

#include "BField.hpp"
#include "RelativisticKinematics.hpp"
#include "Revelator.hpp"
#include <Eigen/Eigen>
#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

struct Particle {
    double m, q;
    Eigen::Vector3d X_t, v_t;
    Eigen::Vector3d u_t; // Proper velocity p/m = gamma*v, in m/s.
    double T_max, dt, hit_prob;
    const double dt_max = 1e-4;

    Particle(double mass, double charge, Eigen::Vector3d position,
             Eigen::Vector3d velocity, double duration, double initial_dt)
        : m(mass), q(charge), X_t(position), v_t(velocity), T_max(duration),
          dt(initial_dt), hit_prob(0.0) {
        if (!std::isfinite(m) || m <= 0.0 || !std::isfinite(q) ||
            !X_t.allFinite() || !v_t.allFinite())
            throw std::invalid_argument("Invalid particle state");
        const double beta = v_t.norm() / c_light;
        if (beta >= 1.0)
            throw std::invalid_argument("Initial velocity must be below light speed");
        u_t = v_t / std::sqrt((1.0 - beta) * (1.0 + beta));
    }

    static Particle fromKineticEnergyMeVPerAmu(
        double mass, double charge, const Eigen::Vector3d& position,
        const Eigen::Vector3d& direction, double energy, double duration, double initial_dt) {
        if (!direction.allFinite() || direction.norm() == 0.0)
            throw std::invalid_argument("Initial direction must be finite and nonzero");
        const auto state = nuclearKinematics(energy, mass);
        Particle particle(mass, charge, position, Eigen::Vector3d::Zero(), duration, initial_dt);
        particle.u_t = state.proper_speed * direction.normalized();
        particle.v_t = velocityFromProper(particle.u_t);
        return particle;
    }

    static double gammaFromProper(const Eigen::Vector3d& u) {
        return std::hypot(1.0, u.norm() / c_light);
    }

    static Eigen::Vector3d velocityFromProper(const Eigen::Vector3d& u) {
        return u / gammaFromProper(u);
    }

    double kineticEnergyEV() const {
        // Stable equivalent of (gamma - 1)*m*c^2/e, including low speeds.
        return m * u_t.squaredNorm() / ((gammaFromProper(u_t) + 1.0) * e_q);
    }

    // Synchronized relativistic Boris for static B and E=0: half drift,
    // rotate proper momentum at the midpoint, half drift (second order).
    template <typename MagneticField>
    void advanceBoris(MagneticField field, Revelator& detector,
                      double max_step = std::numeric_limits<double>::infinity()) {
        v_t = velocityFromProper(u_t);
        dt = selectStep(field(X_t), detector, max_step);
        const Eigen::Vector3d midpoint = X_t + 0.5 * dt * v_t;
        const Eigen::Vector3d B = field(midpoint);
        if (!B.allFinite())
            throw std::runtime_error("Non-finite magnetic field at Boris midpoint");
        const Eigen::Vector3d rotation = (q / m) * (0.5 * dt / gammaFromProper(u_t)) * B;
        const Eigen::Vector3d s = 2.0 * rotation / (1.0 + rotation.squaredNorm());
        const Eigen::Vector3d u_prime = u_t + u_t.cross(rotation);
        const Eigen::Vector3d u_next = u_t + u_prime.cross(s);
        const Eigen::Vector3d X_next = midpoint + 0.5 * dt * velocityFromProper(u_next);
        recordStep(X_next, u_next, detector);
    }

    void updatePosition(BField& field, Revelator& detector,
                        double max_step = std::numeric_limits<double>::infinity()) {
        advanceBoris([&field](const Eigen::Vector3d& x) { return field.totalBField(x); },
                     detector, max_step);
    }

    void updateBS(BField& field, Revelator& detector,
                  double max_step = std::numeric_limits<double>::infinity()) {
        advanceBoris([&field](const Eigen::Vector3d& x) { return field.totalBS(x); },
                     detector, max_step);
    }

    // RK4 reference for the same relativistic equations:
    // dx/dt = v(u), du/dt = (q/m)*v(u) cross B(x).
    template <typename MagneticField>
    void advanceRK4(MagneticField field, Revelator& detector,
                    double max_step = std::numeric_limits<double>::infinity()) {
        v_t = velocityFromProper(u_t);
        dt = selectStep(field(X_t), detector, max_step);
        const Eigen::Vector3d k1_x = v_t;
        const Eigen::Vector3d k1_u = (q / m) * k1_x.cross(field(X_t));
        const Eigen::Vector3d k2_x = velocityFromProper(u_t + 0.5 * dt * k1_u);
        const Eigen::Vector3d k2_u = (q / m) * k2_x.cross(field(X_t + 0.5 * dt * k1_x));
        const Eigen::Vector3d k3_x = velocityFromProper(u_t + 0.5 * dt * k2_u);
        const Eigen::Vector3d k3_u = (q / m) * k3_x.cross(field(X_t + 0.5 * dt * k2_x));
        const Eigen::Vector3d k4_x = velocityFromProper(u_t + dt * k3_u);
        const Eigen::Vector3d k4_u = (q / m) * k4_x.cross(field(X_t + dt * k3_x));
        const Eigen::Vector3d X_next = X_t + (dt / 6.0) * (k1_x + 2*k2_x + 2*k3_x + k4_x);
        const Eigen::Vector3d u_next = u_t + (dt / 6.0) * (k1_u + 2*k2_u + 2*k3_u + k4_u);
        recordStep(X_next, u_next, detector);
    }

    void updatePositionRK4(BField& field, Revelator& detector,
                           double max_step = std::numeric_limits<double>::infinity()) {
        advanceRK4([&field](const Eigen::Vector3d& x) { return field.totalBField(x); },
                   detector, max_step);
    }

    void updatePositionRK4_BS(BField& field, Revelator& detector,
                              double max_step = std::numeric_limits<double>::infinity()) {
        advanceRK4([&field](const Eigen::Vector3d& x) { return field.totalBS(x); },
                   detector, max_step);
    }

private:
    double selectStep(const Eigen::Vector3d& B, const Revelator& detector, double max_step) const {
        if (!B.allFinite() || !std::isfinite(detector.R) || detector.R <= 0.0 ||
            std::isnan(max_step) || max_step <= 0.0)
            throw std::invalid_argument("Invalid field, detector radius or time-step bound");
        const double dt_disp = (detector.R / 4.0) / std::max(v_t.norm(), 1e-9);
        const double Bmag = B.norm();
        const double dt_cycl = (Bmag > 1e-12 && q != 0.0)
            ? 0.1 * 2.0 * PI * gammaFromProper(u_t) * m / (std::abs(q) * Bmag)
            : dt_max;
        // Never enlarge a step past its displacement, gyroperiod or final-time bound.
        const double step = std::min({dt_disp, dt_cycl, dt_max, max_step});
        if (!std::isfinite(step) || step <= 0.0)
            throw std::runtime_error("Unable to select a positive finite time step");
        return step;
    }

    void recordStep(Eigen::Vector3d X_next, const Eigen::Vector3d& u_next, Revelator& detector) {
        hit_prob += 0.5 * dt * (detector.revelatorProbability(X_t) +
                                detector.revelatorProbability(X_next));
        X_t = X_next;
        u_t = u_next;
        v_t = velocityFromProper(u_t);
    }
};
