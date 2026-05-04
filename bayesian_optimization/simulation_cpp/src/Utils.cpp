// Project sources
#include "Utils.hpp"
#include <BField.hpp>
#include <Revelator.hpp>
#include <Particle.hpp>
// External libraries
#include <Eigen/Eigen>
// Starndard libraries for math
#include <vector>
//#include <numeric>
#include "Constants.hpp"
#include <math.h>
#include <cmath>
#include <random>
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
                double& dt, unsigned long& seed, double& expectedDose) {
    // Monte Carlo simulation
    default_random_engine gen;
    uniform_real_distribution<double> azimut(0, 2 * PI);
    uniform_real_distribution<double> polar(0, PI);

    double expectedEVCounter = 0.0; // Counter for how may eV received
    const double conversionEV = 2 * e_q / m;
    for (size_t i = 0; i < N; ++i) {
        // Sample initial position from a sphere of radius 4R
        double theta = azimut(gen);
        double phi = polar(gen);
        Vector3d X0;
        X0 << 50 * sin(phi) * cos(theta), 50 * sin(phi) * sin(theta), 50 * cos(phi);
        // Set target as center of particle detector
        const Vector3d target = - X0;
        const double v_abs = sqrt(energy_samples[i] * conversionEV);
        Vector3d v0 = v_abs * target.normalized();

        // Define particle
        double T_max = 1.5 * target.norm() / v_abs;
        Particle part(m, q, X0, v0, T_max, dt);

        // Start trajectory computation
        double t = 0;
        while (t < T_max) {
            part.updatePosition(field, revelator);
            t += part.dt;
        }
        // Update probablity estimation of hits
        double partHitProb = v_abs * part.hit_prob;
        expectedEVCounter += energy_samples[i] * partHitProb;
    }
    // Return the expected dose computed by this thread.
    expectedDose = expectedEVCounter / static_cast<double>(N);

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

// Thread worker function
void runSimulation(BField field, Revelator revelator, const vector<double>& energy_samples,
                   string name, double m, double q, int N, double dt, unsigned long seed,
                   double& totalExpectedDose, mutex& doseMutex) {
    double expectedDose = 0.0;
    bool ok = monteCarlo(field, revelator, energy_samples, name, m, q, N, dt, seed, expectedDose);

    if (ok) {
        lock_guard<mutex> lock(doseMutex);
        totalExpectedDose += expectedDose;
    }

    {
        lock_guard<mutex> lock(io_mutex);
        if (!ok) {
            cerr << "Seed " << seed << ". Simulation for " << name << " failed.\n";
        }           
    }
}

// Main reader & dispatcher
double runFromCSV_MT(const string& filename, BField field, Revelator revelator,
                     unordered_map<string, vector<double>> samples, int N,
                     double dt, unsigned long seed) {
    ifstream file(filename);
    if (!file.is_open()) {
        cerr << "Seed " << seed << ". Error: Could not open " << filename << endl;
        return -1;
    }
    if (file.peek() == ifstream::traits_type::eof()) {
        cerr << "Seed " << seed << ". Error: File is empty!" << endl;
        return -1;
    }

    string line;
    getline(file, line); // skip header

    vector<thread> threads;
    double totalExpectedDose = 0.0;
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

            // Launch one thread per simulation
            threads.emplace_back(runSimulation, field, revelator, ref(energy_samples), name, m, q, N, dt, seed, ref(totalExpectedDose), ref(doseMutex));
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

    cout << "\n All simulations with seed " << seed << "finished.\n";

    return totalExpectedDose;
}
