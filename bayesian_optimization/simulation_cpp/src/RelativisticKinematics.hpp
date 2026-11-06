#pragma once

#include "Constants.hpp"
#include <cmath>
#include <stdexcept>

struct NuclearKinematics {
    double energy_eV;    // Total kinetic energy of the nucleus, not per amu.
    double proper_speed; // |u| = gamma |v|, in m/s.
    double speed;        // |v|, in m/s.
};

// Source spectra use MeV/amu; the mass table approximates m = A*amu.
inline NuclearKinematics nuclearKinematics(double energy_MeV_per_amu, double mass) {
    if (!std::isfinite(energy_MeV_per_amu) || energy_MeV_per_amu < 0.0 ||
        !std::isfinite(mass) || mass <= 0.0)
        throw std::invalid_argument("Expected nonnegative MeV/amu energy and positive mass");
    const double energy_eV = energy_MeV_per_amu * 1e6 * (mass / amu);
    const double tau = energy_MeV_per_amu * 1e6 * e_q / (amu * c_light * c_light);
    // Avoid cancellation in gamma^2 - 1 at low kinetic energy.
    const double proper_speed = c_light * std::sqrt(tau * (tau + 2.0));
    const double speed = proper_speed / (1.0 + tau);
    if (!std::isfinite(energy_eV) || !std::isfinite(proper_speed))
        throw std::invalid_argument("Kinetic energy exceeds the supported numerical range");
    return {energy_eV, proper_speed, speed};
}
