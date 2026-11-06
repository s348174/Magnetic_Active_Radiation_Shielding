// Project sources
#include "Utils.hpp"
#include <BField.hpp>
#include <Revelator.hpp>
#include <Particle.hpp>
#include <Sobol.hpp>
// External libraries
#include <Eigen/Eigen>
// Starndard libraries for math
#include <vector>
#include "Constants.hpp"
#include <math.h>
#include <cmath>
#include <random>
#include <utility>
// For i/o and string manipulation
#include <iostream>
//#include <iomanip>
#include <sstream>
#include <fstream>
#include <string>
#include <regex>
// For multithreading
#include <thread>
#include <mutex>
#include <functional>
// For directory creation
#include <sys/stat.h>
#include <sys/types.h>
//#include <filesystem>

using namespace std;
using namespace Eigen;

bool monteCarlo(BField& field, Revelator& revelator, const vector<double>& energy_samples,
                const string& particleName, const double& m, const double& q, const int& N,
                double& dt, unsigned long& seed, double& expectedDose, double& variance) {
    // Monte Carlo simulation
    // Sobol sample of initial positions
    uint32_t seed32 = static_cast<uint32_t>(seed ^ (seed >> 32)); // Fold a 64 seed in half
    vector<Vector3d> startPoints = sampleSphereSobol(N, 50, seed32);

    double expectedEVCounter = 0.0; // Counter for how may eV received
    vector<double> energyValues;
    energyValues.reserve(N);
    for (size_t i = 0; i < N; ++i) {
        Vector3d X0 = startPoints[i];
        // Set target as center of particle detector
        const Vector3d target = - X0;
        // Input: MeV/amu. Score: total nuclear kinetic energy in eV.
        const auto kinematics = nuclearKinematics(energy_samples[i], m);
        const double v_abs = kinematics.speed;
        if (v_abs <= 0.0)
            throw invalid_argument("Trajectory energy must be positive");

        // Define particle
        double T_max = 1.5 * target.norm() / v_abs;
        Particle part = Particle::fromKineticEnergyMeVPerAmu(
            m, q, X0, target, energy_samples[i], T_max, dt);

        // Start trajectory computation
        double t = 0;
        while (t < T_max) {
            part.updatePosition(field, revelator, T_max - t);
            t += part.dt;
        }
        // Update probablity estimation of hits
        double partHitProb = v_abs * part.hit_prob;
        double exptEVPart = kinematics.energy_eV * partHitProb;
        expectedEVCounter += exptEVPart;
        energyValues.push_back(exptEVPart);
    }
    // Return the expected dose and the variance computed by this thread.
    expectedDose = expectedEVCounter / static_cast<double>(N);
    double squareDiff = 0.0;
    for (size_t i = 0; i < energyValues.size(); ++i) {
        squareDiff += (energyValues[i] - expectedDose) * (energyValues[i] - expectedDose);
    }
    variance = squareDiff / static_cast<double>(N - 1);

    return true;
}

// Thread-safe printing
mutex io_mutex;

static inline string trim(const string& s) {
    string out = s;
    out.erase(0, out.find_first_not_of(" \t\r\n"));
    out.erase(out.find_last_not_of(" \t\r\n") + 1);
    return out;
}

// Evaluate simple expressions like "4*1.673e-27" or plain "9.1e-31"
double evaluateExpression(const string& expr) {
    regex pattern(R"(([0-9\.eE\+\-]+)\*([0-9\.eE\+\-]+))");
    smatch match;
    if (std::regex_match(expr, match, pattern)) {
        return std::stod(match[1].str()) * std::stod(match[2].str());
    }
    return std::stod(expr);
}

// Integrate each original spectrum before its normalization as a sampling PDF.
unordered_map<string, double> readFluxWeights(const string& filename) {
    ifstream file(filename);
    if (!file.is_open()) throw runtime_error("Could not open flux file: " + filename);

    string line, name;
    if (!getline(file, line)) throw invalid_argument("Flux file is empty");
    stringstream header(line);
    getline(header, name, ',');
    if (trim(name) != "Energy") throw invalid_argument("Flux file must start with Energy");

    vector<string> names;
    unordered_map<string, double> weights;
    while (getline(header, name, ',')) {
        name = trim(name);
        if (name.empty() || weights.count(name)) throw invalid_argument("Invalid flux species name");
        names.push_back(name);
        weights[name] = 0.0;
    }
    if (names.empty()) throw invalid_argument("Flux file has no particle species");

    vector<double> previousFlux(names.size());
    double previousEnergy = 0.0;
    int rows = 0;
    while (getline(file, line)) {
        if (trim(line).empty()) continue;
        stringstream ss(line);
        string value;
        getline(ss, value, ',');
        double energy = stod(value);
        if (!isfinite(energy) || energy < 0.0 || (rows > 0 && energy <= previousEnergy))
            throw invalid_argument("Flux energies must be finite and strictly increasing");
        for (size_t i = 0; i < names.size(); ++i) {
            if (!getline(ss, value, ',')) throw invalid_argument("Missing flux value");
            double flux = stod(value);
            if (!isfinite(flux) || flux < 0.0) throw invalid_argument("Invalid flux value");
            if (rows > 0)
                weights[names[i]] += 0.5 * (previousFlux[i] + flux) * (energy - previousEnergy);
            previousFlux[i] = flux;
        }
        if (getline(ss, value, ',')) throw invalid_argument("Extra flux column");
        previousEnergy = energy;
        ++rows;
    }
    if (rows < 2) throw invalid_argument("Flux file needs at least two energy rows");
    for (const auto& entry : weights)
        if (!isfinite(entry.second)) throw invalid_argument("Non-finite integrated flux");
    return weights;
}

// Thread worker function
void runSimulation(BField field, Revelator revelator, const vector<double>& energy_samples,
                   string name, double m, double q, double fluxWeight, int N, double dt, unsigned long seed,
                   double& totalExpectedDose, double& totalVariance, mutex& doseMutex) {
    double expectedDose = 0.0;
    double variance = 0.0;
    bool ok = monteCarlo(field, revelator, energy_samples, name, m, q, N, dt, seed, expectedDose, variance);

    if (ok) {
        lock_guard<mutex> lock(doseMutex);
        totalExpectedDose += fluxWeight * expectedDose;
        totalVariance += fluxWeight * fluxWeight * variance / static_cast<double>(N);
    }

    {
        lock_guard<mutex> lock(io_mutex);
        if (!ok) {
            cerr << "Seed " << seed << ". Simulation for " << name << " failed.\n";
        }           
    }
}

// Main reader & dispatcher
pair<double, double> runFromCSV_MT(const string& filename, BField field, Revelator revelator,
                     unordered_map<string, vector<double>> samples, int N,
                     double dt, unsigned long seed) {
    ifstream file(filename);
    if (!file.is_open()) {
        cerr << "Seed " << seed << ". Error: Could not open " << filename << endl;
        return make_pair(-1, -1);
    }
    if (file.peek() == ifstream::traits_type::eof()) {
        cerr << "Seed " << seed << ". Error: File is empty!" << endl;
        return make_pair(-1, -1);
    }

    string line;
    getline(file, line); // skip header

    // Restore physical species abundances, not the visualization-scaled fluxes.
    const unordered_map<string, double> fluxWeights = readFluxWeights("../data/flux_data_clean.csv");
    for (const auto& entry : samples) {
        if (!fluxWeights.count(entry.first))
            throw invalid_argument("Missing integrated flux for " + entry.first);
    }

    vector<thread> threads;
    double totalExpectedDose = 0.0;
    double totalVariance = 0.0;
    mutex doseMutex;

    cout << "Starting simulations with seed " << seed << "..." << endl;

    while (getline(file, line)) {

        if (line.empty()) continue;

        stringstream ss(line);
        string idxStr, name, mStr, qStr;
        getline(ss, idxStr, ',');
        getline(ss, name, ',');
        getline(ss, mStr, ',');
        getline(ss, qStr, ',');

        mStr = trim(mStr);
        qStr = trim(qStr);

        try {
            double m = amu * evaluateExpression(mStr);
            double q = e_q * evaluateExpression(qStr);
            const vector<double>& energy_samples = samples.at(name);
            double fluxWeight = fluxWeights.at(name);

            // Launch one thread per simulation
            threads.emplace_back(runSimulation, field, revelator, ref(energy_samples), name, m, q, fluxWeight, N, dt, seed, ref(totalExpectedDose), ref(totalVariance), ref(doseMutex));
        }
        catch (const std::exception& e) {
            lock_guard<mutex> lock(io_mutex);
            cerr << "Seed " << seed << ". Error parsing line: " << line << "\n" << e.what() << endl;
        }
    }

    file.close();

    // Join all threads
    for (auto& th : threads) {
        if (th.joinable()) th.join();
    }

    cout << "\n All simulations with seed " << seed << " finished.\n";

    return make_pair(totalExpectedDose, totalVariance);
}
