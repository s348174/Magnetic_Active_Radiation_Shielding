#pragma once

#include <Eigen/Eigen>
#include <vector>
#include <BField.hpp>
#include <Revelator.hpp>
#include <string>
#include <mutex>

using namespace std;
using namespace Eigen;

double mbPdf(const double v, const double m, const double T);
vector<double> sampleMbSpeed(const double m, const int N, const double T);
bool monteCarlo(BField& field, Revelator& revelator, const string& particleName, const double& m, const double& q, const int& N, const double& T, double& dt, unsigned long& seed, double& expectedDose);
double evaluateExpression(const string& expr);
void runSimulation(BField field, Revelator revelator, string name, double m, double q, int N, double T, double dt, unsigned long seed, double& totalExpectedDose, mutex& doseMutex);
double runFromCSV_MT(const string& filename, BField field, Revelator revelator, int N, double T, double dt, unsigned long seed);
