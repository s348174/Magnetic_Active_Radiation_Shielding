#pragma once

#include <vector>
#include <Eigen/Dense>

std::vector<Eigen::Vector3d> sampleSphereSobol(int N, double R, uint32_t seed);
