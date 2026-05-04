#include "Test.hpp"
#include <Eigen/Eigen>
#include <iostream>
#include <Revelator.hpp>
#include <BField.hpp>

using namespace std;
using namespace Eigen;

int main()
{
    cout << "Starting tests..." << endl;
    vector<Vector3d> positions;
    vector<Vector3d> speeds;
    // Test 1: Single particle with initial position (50,0,0) and velocity (0,1e6,0)
    cout << "Test 1: Single particle with initial position (50,0,0) and velocity (0,1e6,0)" << endl;
    positions.push_back(Vector3d(50, 0, 0));
    speeds.push_back(Vector3d(0, 1e6, 0));
    computeMetrics(positions, speeds);
    // Test2: Multiple particles with random initial positions and random initial velocities in range [1e6-1e8]
    cout << "Test 2: Multiple particles with random initial positions and random initial velocities in range [1e6-1e8]" << endl;
    positions.clear();
    speeds.clear();
    for (int i = 0; i < 100; ++i) {
        Vector3d pos = Vector3d::Random() * 50; // Random position
        double v0 = 1e6 + (1e8 - 1e6) * (rand() / (double)RAND_MAX); // Random speed norm [1e6-1e8]
        Vector3d direction = -pos.normalized(); // Direction from pos towards origin
        Vector3d vel = direction * v0; // Velocity vector
        positions.push_back(pos);
        speeds.push_back(vel);
    }
    computeMetrics(positions, speeds);
    return 0;
}
