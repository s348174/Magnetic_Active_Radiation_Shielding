#pragma once

#include <Eigen/Eigen>
#include <vector>

using namespace std;
using namespace Eigen;

double MB_pdf(const double v, const double m, const double kB, const double T);
vector<double> sample_MB_speed(const double m, const int N);

