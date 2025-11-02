#pragma once

#include <Eigen/Eigen>
#include <vector>

using namespace std;
using namespace Eigen;

struct Trajectory {
    vector<Vector3d> X; // Positions
    vector<Vector3d> v; // Speeds
    vector<Vector3d> a; // Accelerations
    vector<Vector3d> p; // Relativistic momentum
};
