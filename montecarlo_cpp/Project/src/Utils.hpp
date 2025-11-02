#pragma once

#include <Eigen/Eigen>
#include <vector>
#include <Torus.hpp>
#include <string>

using namespace std;
using namespace Eigen;

double mbPdf(const double v, const double m, const double kB, const double T);
vector<double> sampleMbSpeed(const double m, const int N, const double T);
bool monteCarlo(Torus& torus, const string& particleName, const double& m, const double& q, const int& N, const double& T, const double& dt, unsigned long& seed);
double evaluateExpression(const string& expr);
void runSimulation(Torus torus, string name, double m, double q, int N, double T, double dt, unsigned long seed);
void runFromCSV_MT(const string& filename, Torus torus, int N, double T, double dt);
