#include "Utils.hpp"
#include <vector>
#include <Eigen/Eigen>
#include <numbers>
#include <math.h>
#include <cmath>
#include <random>
#include <Torus.hpp>
#include <Particle.hpp>
#include <iostream>
#include <iomanip>
#include <sstream>
#include <fstream>
#include <string>
#include <thread>
#include <mutex>
#include <regex>
#include <sys/stat.h>
#include <sys/types.h>
#include <filesystem>

using namespace std;
using namespace Eigen;

double mbPdf(const double v, const double m, const double kB, const double T)
{
    // Maxwell-Boltzman pdf: f(v) ∝ v^2 * exp(-v^2)
    double f = pow(m / (2 * M_PI * kB * T), 1.5)
               * 4 * M_PI * v * v * exp(- m * v * v / ( 2 * kB * T));
    return f;
}

vector<double> sampleMbSpeed(const double m, const int N, const double T)
{
    // Boltzmann constand and Sun's photosphere temperature
    const double kB = 1.38e-23;   // J/K

    // Sampling interval
    const double v_min = 0;
    const double v_max = 1e7;

    // Define f_max
    double v_mean = sqrt((2*kB*T)/m);
    double f_max = mbPdf(v_mean, m, kB, T);

    // Define a vector of N samples
    vector<double> v_samples;
    v_samples.reserve(N);

    // Accept/rejept loop
    int i = 0;
    default_random_engine gen;
    uniform_real_distribution<double> velocity(v_min, v_max);
    uniform_real_distribution<double> density(0, f_max);
    while (i < N) {
        double v_rand = velocity(gen);
        double y_rand = density(gen);
        if (y_rand <= mbPdf(v_rand, m, kB, T)) { // Acceptance region
            v_samples.push_back(v_rand);
            i++;
        }
    }
    return v_samples;
}

bool monteCarlo(Torus& torus, const string& particleName, const double& m, const double& q, const int& N, const double& T, const double& dt) { // Monte Carlo simulation
    // Sample speeds from Maxwell Boltzman
    vector<double> v_samples = sampleMbSpeed(m, N, T);
    sort(v_samples.begin(), v_samples.end());
    default_random_engine gen;
    uniform_real_distribution<double> azimut(0, 2 * M_PI);
    uniform_real_distribution<double> polar(0, M_PI);

    // Open file for output
    string folderout = "results";
    if (mkdir(folderout.c_str(), 0777) == -1) {
        if (errno != EEXIST) {
            cerr << "Could not create directory " << folderout << endl;
        }
    }
    ostringstream filename;
    filename << folderout << particleName << "_results.csv";
    ofstream outfile(filename.str());
    if (!outfile.is_open()) {
        cerr << "Error: could not open output file.\n";
        return false;
    }

    outfile << setprecision(6);
    outfile << "i,eV,hit_status,x_0,y_0,z_0,v_x,v_y,v_z\n"; // CSV header

    int hitCounter = 0; // Counter for how many times the torus gets hit
    for (size_t i = 0; i < v_samples.size(); ++i) {
        // Sample initial position from a sphere of radius 4R
        double theta = azimut(gen);
        double phi = polar(gen);
        Vector3d X0;
        X0 << 4 * torus.R * sin(phi) * cos(theta), 4 * torus.R * sin(phi) * sin(theta), 4 * torus.R * cos(phi);

        // Sample random target inside torus
        double alpha = azimut(gen);
        Vector3d target(torus.R * cos(alpha), torus.R * sin(alpha), 0);
        target = target - X0;
        Vector3d v0 = v_samples[i] * target / target.norm();

        // Define particle
        double T_max = 1.5 * target.norm() / v_samples[i];
        Particle part(m, q, X0, v0, T_max, dt);

        // Start trajectory computation
        double t = 0;
        while (!part.updatePosition(torus) && t < T_max) {
            t += dt;
        }
        if (part.hit) {
            hitCounter++;
            outfile << i << "," << scientific << 0.5 * m * v_samples[i] * v_samples[i] * 1.6022e-19 << ",hit,"
                    << fixed << X0(0) << "," << X0(1) << "," << X0(2) << ","
                    << scientific << v0(0) << "," << v0(1) << "," << v0(2) << "\n";
        }
        else {
            outfile << i << "," << scientific << 0.5 * m * v_samples[i] * v_samples[i] * 1.6022e-19 << ",miss,"
                    << fixed << X0(0) << "," << X0(1) << "," << X0(2) << ","
                    << scientific << v0(0) << "," << v0(1) << "," << v0(2) << "\n";
        }
    }
    double hitRatio = hitCounter / static_cast<double>(N);
    cout << "There have been " << hitCounter << " hits" << endl;
    cout << "Your hit ratio is " << hitRatio << endl;
    double successPerc = 100 * (1 - hitRatio);
    cout << "Your success percentage is: " << successPerc << "%" << endl;

    // Write summary to file
    outfile << "\nSummary\n";
    outfile << fixed << "Radius," << torus.R << "m\n";
    outfile << scientific << setprecision(4);
    outfile << "Current," << torus.I << "A\n";
    outfile << "Mass," << m << "kg\n";
    outfile << "Charge," << q << "C\n";
    outfile << fixed;
    outfile << "Total," << N << "\n";
    outfile << "Hits," << hitCounter << "\n";
    outfile << "Hit ratio," << hitRatio << "\n";
    outfile << "Success percentage," << successPerc << "%\n";

    outfile.close();

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
void runSimulation(Torus torus, string name, double m, double q, int N, double T, double dt) {
    {
        lock_guard<mutex> lock(io_mutex);
        cout << "\n=== Starting simulation for " << name << " ===" << endl;
        cout << "m = " << m << ", q = " << q << endl;
    }

    bool ok = monteCarlo(torus, name, m, q, N, T, dt);

    {
        lock_guard<mutex> lock(io_mutex);
        if (ok)
            cout << "Simulation for " << name << " completed.\n";
        else
            cerr << "Simulation for " << name << " failed.\n";
    }
}

// Main reader & dispatcher
void runFromCSV_MT(const string& filename, Torus torus, int N, double T, double dt) {
    ifstream file(filename);
    if (!file.is_open()) {
        cerr << "Error: Could not open " << filename << endl;
        return;
    }
    if (file.peek() == ifstream::traits_type::eof()) {
        cerr << "Error: File is empty!" << endl;
        return;
    }

    string line;
    getline(file, line); // skip header
    cout << line << endl;

    vector<thread> threads;

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
            double m = evaluateExpression(mStr);
            double q = evaluateExpression(qStr);

            // Launch one thread per simulation
            cout << "Launching thread for " << name << " (m=" << m << ", q=" << q << ")" << endl;
            threads.emplace_back(runSimulation, torus, name, m, q, N, T, dt);
        }
        catch (const std::exception& e) {
            lock_guard<mutex> lock(io_mutex);
            cerr << "Error parsing line: " << line << "\n" << e.what() << endl;
        }


    }

    file.close();

    // Join all threads
    for (auto& th : threads) {
        if (th.joinable()) th.join();
    }

    cout << "\n All simulations finished.\n";
}
