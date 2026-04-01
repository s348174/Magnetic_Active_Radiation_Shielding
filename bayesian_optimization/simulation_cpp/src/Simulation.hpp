#pragma once

#include <Eigen/Eigen>
#include <vector>

using namespace std;
using namespace Eigen;

double launch_simulation(unsigned long seed, int K, double I, double R, vector<Vector3d> centers, vector<Vector3d> normals);
