#include "Utils.hpp"
#include <Eigen/Eigen>
#include <iostream>

using namespace std;
using namespace Eigen;

int main()
{
    cout << a << endl;
    double m = 1.67e-27;
    double N = 100;
    vector<double> v_samples = sampleMbSpeed(m, N);
    for (size_t i = 0; i < v_samples.size(); ++i) {
        cout << v_samples[i] << endl;
    }
    return 0;
}
