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

bool monteCarlo(BField& field, Revelator& revelator, const vector<double>& energy_samples,
                const string& particleName, const double& m, const double& q, const int& N,
                double& dt, unsigned long& seed, double& expectedDose, double& variance);
double evaluateExpression(const string& expr);
void runSimulation(BField field, Revelator revelator, const vector<double>& energy_samples,
                   string name, double m, double q, int N, double dt, unsigned long seed,
                   double& totalExpectedDose, double& totalVariance, mutex& doseMutex);
pair<double, double> runFromCSV_MT(const string& filename, BField field, Revelator revelator,
                     unordered_map<string, vector<double>> samples, int N,
                     double dt, unsigned long seed);
