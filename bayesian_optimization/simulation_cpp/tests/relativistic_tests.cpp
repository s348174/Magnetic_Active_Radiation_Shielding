#include "Particle.hpp"
#include "Utils.hpp"
#include <array>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>

using Eigen::Vector3d;

void require(bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message);
}

void testKinematics() {
    for (double energy : {1e-8, 0.01, 1.0, 1000.0, 1e6}) {
        const auto proton = nuclearKinematics(energy, amu);
        for (double A : {1.0, 4.0, 56.0}) {
            const auto state = nuclearKinematics(energy, A * amu);
            require(std::abs(state.energy_eV / (A * energy * 1e6) - 1.0) < 1e-14,
                    "MeV/amu to total eV conversion");
            require(state.speed > 0.0 && state.speed < c_light, "Subluminal initial speed");
            require(std::abs(state.speed / proton.speed - 1.0) < 1e-14,
                    "Equal energy per amu must give equal nuclear speeds");
            Particle p = Particle::fromKineticEnergyMeVPerAmu(
                A*amu, e_q, Vector3d::Zero(), Vector3d(1, 0, 0), energy, 1.0, 1e-9);
            require(std::abs(p.kineticEnergyEV() / state.energy_eV - 1.0) < 1e-12,
                    "Momentum initialization must retain kinetic energy");
        }
    }
    const double tiny = 1e-8;
    const double classical = std::sqrt(2 * tiny * 1e6 * e_q / amu);
    require(std::abs(nuclearKinematics(tiny, amu).speed / classical - 1.0) < 1e-10,
            "Nonrelativistic limit");
    bool rejected = false;
    try { nuclearKinematics(-1.0, amu); } catch (const std::invalid_argument&) { rejected = true; }
    require(rejected, "Negative kinetic energy must be rejected");
    rejected = false;
    try { Particle(amu, e_q, Vector3d::Zero(), Vector3d(c_light, 0, 0), 1, 1e-9); }
    catch (const std::invalid_argument&) { rejected = true; }
    require(rejected, "Superluminal initial velocity must be rejected");
    std::cout << "PASS energy units, mass scaling, gamma and low-energy limit\n";
}

void testDrift() {
    const auto zero = [](const Vector3d&) { return Vector3d::Zero().eval(); };
    Revelator detector(100);
    Particle p = Particle::fromKineticEnergyMeVPerAmu(
        amu, e_q, Vector3d(1,2,3), Vector3d(1,2,-1), 1e6, 1, 1e-9);
    const Vector3d start = p.X_t, velocity = p.v_t, momentum = p.u_t;
    const double h = 1e-10;
    for (int i = 0; i < 100; ++i) p.advanceBoris(zero, detector, h);
    require((p.X_t - start - 100*h*velocity).norm() < 1e-12, "Zero-field straight-line motion");
    require((p.u_t - momentum).norm() == 0.0, "Zero-field momentum conservation");
    p.advanceBoris(zero, detector, 1e-13);
    require(p.dt <= 1e-13, "Final-time bound must not be raised to an artificial minimum");
    std::cout << "PASS zero-field drift and short final step\n";
}

double helixError(int n, double charge, double energy, bool rk4 = false) {
    const auto field = [](const Vector3d&) { return Vector3d(0,0,1); };
    Revelator detector(1e6); // Keep displacement control above the prescribed refinement steps.
    Particle p = Particle::fromKineticEnergyMeVPerAmu(
        amu, charge, Vector3d::Zero(), Vector3d(.8,0,.6), energy, 1, 1e-9);
    const Vector3d v0 = p.v_t;
    const double gamma = Particle::gammaFromProper(p.u_t);
    const double initial_energy = p.kineticEnergyEV();
    const double omega = charge / (gamma * amu);
    const double duration = .7 / std::abs(omega), h = duration/n;
    for (int i = 0; i < n; ++i) {
        if (rk4) p.advanceRK4(field, detector, h);
        else p.advanceBoris(field, detector, h);
    }
    const double theta = omega*duration;
    const Vector3d exact(v0.x()*std::sin(theta)/omega,
                         v0.x()*(std::cos(theta)-1)/omega, v0.z()*duration);
    if (!rk4) {
        require(std::abs(p.kineticEnergyEV()/initial_energy - 1.0) < 1e-11,
                "Boris kinetic energy conservation");
        require(std::abs(p.v_t.norm()/v0.norm() - 1.0) < 1e-12, "Boris speed conservation");
    }
    return (p.X_t-exact).norm() / (v0.norm()*duration);
}

void testHelices() {
    for (double energy : {.01, 1000.0, 1e6}) {
        for (double charge : {-e_q, e_q}) {
            double previous = helixError(20, charge, energy);
            for (int n : {40,80,160}) {
                const double error = helixError(n, charge, energy);
                require(previous/error > 3.8 && previous/error < 4.2,
                        "Relativistic Boris position must converge at second order");
                previous = error;
            }
        }
    }
    const double coarse = helixError(20, e_q, 1000, true);
    const double fine = helixError(40, e_q, 1000, true);
    require(coarse/fine > 14 && coarse/fine < 18, "Relativistic RK4 reference order");
    std::cout << "PASS analytic helices, both charge signs, second-order Boris and fourth-order RK4\n";
}

void testNonuniformField() {
    BField coil(1, {Vector3d::Zero()}, {Vector3d(0,0,1)}, .25, 72000);
    Revelator detector(2);
    const double duration = 5e-8;
    auto initial = [&]() {
        return Particle::fromKineticEnergyMeVPerAmu(
            amu, e_q, Vector3d(.05,.02,.1), Vector3d(1,.2,-.1), .01, duration, 1e-9);
    };
    Particle reference = initial();
    for (int i=0; i<4096; ++i) reference.updatePositionRK4(coil, detector, duration/4096);
    double previous = 0;
    for (int n : {20,40,80,160}) {
        Particle p = initial();
        for (int i=0; i<n; ++i) p.updatePosition(coil, detector, duration/n);
        const double error = (p.X_t-reference.X_t).norm();
        if (previous > 0) require(previous/error > 3.7 && previous/error < 4.3,
                                  "Second-order position convergence in nonuniform coil field");
        std::cout << "  nonuniform field n=" << n << " position error=" << error << '\n';
        previous = error;
    }
    std::cout << "PASS midpoint field evaluation in nonuniform field\n";
}

void testSimulatorEnergyScore() {
    BField field(1, {Vector3d::Zero()}, {Vector3d(0,0,1)}, .25, 0.0);
    Revelator detector(2);
    const int N = 16;
    const std::vector<double> energies(N, 1.0); // 1 MeV/amu, for each species.
    unsigned long seed = 67;
    double dt = 1e-9, proton_mean, proton_var, iron_mean, iron_var;
    monteCarlo(field, detector, energies, "proton", amu, e_q, N, dt, seed, proton_mean, proton_var);
    monteCarlo(field, detector, energies, "Fe56", 56*amu, 26*e_q, N, dt, seed, iron_mean, iron_var);
    require(std::abs(iron_mean/proton_mean-56.0) < 1e-10, "Total-energy response must scale with nuclear mass");
    // Central straight lines have v*integral rho dt = 1 up to detector quadrature error.
    require(std::abs(proton_mean/1e6 - 1.0) < .01, "Zero-field score must use total eV");
    require(std::isfinite(proton_var) && std::isfinite(iron_var), "Finite response variances");
    std::cout << "PASS simulator energy scoring: proton=" << proton_mean << " eV, iron=" << iron_mean << " eV\n";
}

void testFluxWeights() {
    const std::string filename = (std::filesystem::temp_directory_path() / "mars_flux_weights_test.csv").string();
    {
        std::ofstream file(filename);
        file << "Energy,proton,Fe56\n1,2,10\n3,4,20\n6,2,10\n";
    }
    const auto weights = readFluxWeights(filename);
    std::filesystem::remove(filename);
    require(weights.size() == 2, "Read one integrated flux per species");
    require(std::abs(weights.at("proton") - 15.0) < 1e-12,
            "Trapezoidal integration on nonuniform energy grid");
    require(std::abs(weights.at("Fe56") - 75.0) < 1e-12,
            "Preserve relative abundance without normalizing species weights");
    {
        std::ofstream file(filename);
        file << "Energy,proton\n1,2\n1,3\n";
    }
    bool rejected = false;
    try { readFluxWeights(filename); } catch (const std::invalid_argument&) { rejected = true; }
    std::filesystem::remove(filename);
    require(rejected, "Reject a non-increasing flux energy grid");
    {
        std::ofstream file(filename);
        file << "Energy,proton\n1,2\n2,-3\n";
    }
    rejected = false;
    try { readFluxWeights(filename); } catch (const std::invalid_argument&) { rejected = true; }
    std::filesystem::remove(filename);
    require(rejected, "Reject negative differential flux");

    BField field(1, {Vector3d::Zero()}, {Vector3d(0,0,1)}, .25, 0.0);
    Revelator detector(2);
    const std::vector<double> energies = {.1,.2,.5,1,2,3,5,8};
    const int N = static_cast<int>(energies.size());
    unsigned long seed = 67;
    double dt = 1e-9, proton_mean, proton_var, iron_mean, iron_var;
    monteCarlo(field, detector, energies, "proton", amu, e_q, N, dt, seed, proton_mean, proton_var);
    monteCarlo(field, detector, energies, "Fe56", 56*amu, 26*e_q, N, dt, seed, iron_mean, iron_var);
    double total = 0.0, variance = 0.0;
    std::mutex doseMutex;
    runSimulation(field, detector, energies, "proton", amu, e_q, weights.at("proton"),
                  N, dt, seed, total, variance, doseMutex);
    runSimulation(field, detector, energies, "Fe56", 56*amu, 26*e_q, weights.at("Fe56"),
                  N, dt, seed, total, variance, doseMutex);
    const double expected = 15*proton_mean + 75*iron_mean;
    const double expected_variance = (15*15*proton_var + 75*75*iron_var)/N;
    require(proton_var > 0.0 && iron_var > 0.0, "Nonzero variance in weighting regression");
    require(std::abs(total/expected - 1.0) < 1e-12, "Flux-weighted sum of species means");
    require(std::abs(variance/expected_variance - 1.0) < 1e-12,
            "Species variance must use squared flux weights and divide by N");
    total = variance = 0.0;
    runSimulation(field, detector, energies, "proton", amu, e_q, 0.0,
                  N, dt, seed, total, variance, doseMutex);
    require(total == 0.0 && variance == 0.0, "Zero flux gives zero mean and variance contribution");
    std::cout << "PASS physical flux integration, species-weighted response and squared-weight variance\n";
}

// Independent midpoint quadrature of the filament Biot-Savart law, outside the wire.
Vector3d referenceCoilField(const Vector3d& X, double R, double I) {
    const int steps = 20000;
    const double h = 2 * PI / steps;
    Vector3d sum = Vector3d::Zero();
    for (int i = 0; i < steps; ++i) {
        double angle = (i + 0.5) * h;
        Vector3d source(R*cos(angle), R*sin(angle), 0);
        Vector3d tangent(-R*sin(angle), R*cos(angle), 0);
        Vector3d distance = X - source;
        sum += tangent.cross(distance) / pow(distance.norm(), 3);
    }
    return mu0 * I * h / (4 * PI) * sum;
}

void testFiniteWireCore() {
    const double R = .25, I = 72000, a = BField::wireRadius;
    BField field(1, {Vector3d::Zero()}, {Vector3d(0,0,1)}, R, I);
    const double surfaceField = mu0 * I / (2 * PI * a);
    for (double angle : {0.0, PI/3, PI}) {
        const Vector3d radial(cos(angle), sin(angle), 0);
        const Vector3d tangent(-sin(angle), cos(angle), 0);
        const std::vector<Vector3d> offsets = {Vector3d::Zero(), .5*a*radial,
            -.5*a*radial, Vector3d(0,0,.5*a), Vector3d(0,0,-.5*a),
            .3*a*radial + Vector3d(0,0,.4*a)};
        for (const Vector3d& offset : offsets) {
            const Vector3d X = R*radial + offset;
            const Vector3d expected = surfaceField/a * tangent.cross(offset);
            const Vector3d sam = field.computeSAM(X), bs = field.computeBS(X);
            require(sam.allFinite() && bs.allFinite(), "Finite field inside conductor, including centreline");
            require((sam-expected).norm() < 1e-10*surfaceField, "Straight-wire interior field and direction");
            require((bs-sam).norm() < 1e-12*surfaceField, "Both field evaluators share the same wire core");
        }
    }
    // Exactly on the boundary and just outside it, retain the original circular-coil field.
    for (const Vector3d& X : {Vector3d(R,0,a), Vector3d(R+1.01*a,0,0), Vector3d(.1,.2,.3)}) {
        const Vector3d expected = referenceCoilField(X,R,I);
        const Vector3d actual = field.computeSAM(X);
        require(actual.allFinite(), "Finite exterior field near the wire surface");
        require((actual-expected).norm() / expected.norm() < 1e-8, "Exterior elliptic field remains unchanged");
    }
    const Vector3d centerField = field.computeSAM(Vector3d::Zero());
    require(std::abs(centerField.z() / (mu0*I/(2*R)) - 1.0) < 1e-14,
            "Field in coil aperture retains its analytic axis value");

    BField rotated(1, {Vector3d(1,2,3)}, {Vector3d(1,0,0)}, R, I);
    const Vector3d X = Vector3d(1,2,3) + Vector3d(.5*a,0,R);
    const Vector3d expected(0,0,.5*surfaceField);
    require((rotated.totalBField(X)-expected).norm() < 1e-10*surfaceField,
            "Wire-core field rotates and translates with the coil");
    require((rotated.totalBS(X)-expected).norm() < 1e-10*surfaceField,
            "Biot-Savart comparison uses the same rotated conductor core");
    std::cout << "PASS 1 mm conductor core, centreline, orientations and unchanged exterior field\n";
}

int main() {
    try {
        testKinematics();
        testDrift();
        testHelices();
        testNonuniformField();
        testSimulatorEnergyScore();
        testFluxWeights();
        testFiniteWireCore();
    } catch (const std::exception& error) {
        std::cerr << "FAIL: " << error.what() << '\n';
        return 1;
    }
    return 0;
}
