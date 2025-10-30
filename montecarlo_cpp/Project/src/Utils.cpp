#include "Utils.hpp"
#include <vector>
#include <Eigen/Eigen>
#include <numbers>
#include <math.h>
#include <cmath>
#include <random>

using namespace std;
using namespace Eigen;

double mbPdf(const double v, const double m, const double kB, const double T)
{
    // Maxwell-Boltzman pdf: f(v) ∝ v^2 * exp(-v^2)
    double f = pow(m/(2*M_PI*kB*T),3/2)*4*M_PI*pow(v,2)*exp(-(m*pow(v,2))/(2*kB*T));
    return f;
}

vector<double> sampleMbSpeed(const double m, const int N)
{
    // Boltzmann constand and Sun's photosphere temperature
    const double kB = 1.38e-23;   // J/K
    const double T = 5800;        // K

    // Sampling interval
    const double v_min = 0;
    const double v_max = 30000;

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
    while (i <= N) {
        double v_rand = velocity(gen);
        double y_rand = density(gen);
        if (y_rand <= mbPdf(v_rand, m, kB, T)) { // Acceptance region
            v_samples.push_back(v_rand);
            i += 1;
        }
    }
    return v_samples;
}
