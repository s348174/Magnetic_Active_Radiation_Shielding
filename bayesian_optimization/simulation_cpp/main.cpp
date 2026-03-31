#include "Utils.hpp"
#include <Eigen/Eigen>
#include <iostream>
#include <Revelator.hpp>
#include <BField.hpp>
#include <Utils.hpp>
#include <vector>
#include <chrono>

using namespace std;
using namespace Eigen;

int main()
{
    chrono::steady_clock::time_point t_begin = chrono::steady_clock::now();
    // Simulation arguments
    double N = 10; // Number of simulated particles
    const double T = 1e7; // K
    double dt = 1e-9; // Initial time step
    unsigned long seed = 30006;

    // Define Revelator
    double rho = 10;
    Revelator revelator(rho);

    // Define B Field generator
    double K = 1;
    double I = 0;
    double R = 15;
    vector<Vector3d> centers;
    Vector3d origin;
    origin << 0, 0, 0;
    centers.push_back(origin);
    vector<Vector3d> normals;
    Vector3d z;
    z << 0, 0, 1;
    normals.push_back(z);
    BField field(K, centers, normals, R, I);

    // Run multithread simulation from CSV input (for particles phyisics data)
    double totalExpectedDose = runFromCSV_MT("../simulation_cpp/particles_input.csv", field, revelator, N, T, dt, seed);

    chrono::steady_clock::time_point t_end = chrono::steady_clock::now();
    double elapsedTime = chrono::duration_cast<chrono::milliseconds>(t_end-t_begin).count();
    cout << "Elapsed simulation time: " << elapsedTime << "ms." << endl;
    cout << "Total expected dose: " << totalExpectedDose << endl;
    return 0;
}
