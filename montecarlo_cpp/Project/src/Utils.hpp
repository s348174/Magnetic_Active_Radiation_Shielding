#pragma once

#include <Eigen/Eigen>
#include <vector>
#include <Torus.hpp>

using namespace std;
using namespace Eigen;

double mbPdf(const double v, const double m, const double kB, const double T);
vector<double> sampleMbSpeed(const double m, const int N);
bool monteCarlo(Torus& torus, const double& m, const double& q, const int& N, const double& dt);
