#pragma once

#include <Eigen/Eigen>
#include <vector>
#include <BField.hpp>
#include <Revelator.hpp>
#include <string>
#include <mutex>
#include <unordered_map>

using namespace std;
using namespace Eigen;

// energy_samples: kinetic energy in MeV/amu; expectedDose: energy-weighted score in eV.
bool monteCarlo(BField& field, Revelator& revelator, const vector<double>& energy_samples,
                const string& particleName, const double& m, const double& q, const int& N,
                double& dt, unsigned long& seed, double& expectedDose, double& variance);
double evaluateExpression(const string& expr);
// Integrate physical differential flux over energy (MeV/amu) for each species.
unordered_map<string, double> readFluxWeights(const string& filename);
void runSimulation(BField field, Revelator revelator, const vector<double>& energy_samples,
                   string name, double m, double q, double fluxWeight, int N, double dt, unsigned long seed,
                   double& totalExpectedDose, double& totalVariance, mutex& doseMutex);
// Returns the flux-weighted response and variance; species weights come from
// ../data/flux_data_clean.csv, in particles/(day*cm^2), without normalization.
pair<double, double> runFromCSV_MT(const string& filename, BField field, Revelator revelator,
                     unordered_map<string, vector<double>> samples, int N,
                     double dt, unsigned long seed);
