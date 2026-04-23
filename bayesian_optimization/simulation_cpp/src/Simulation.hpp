#pragma once

#include <Eigen/Eigen>
#include <vector>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

using namespace std;
using namespace Eigen;
namespace py = pybind11;

std::vector<Eigen::Vector3d> numpy_to_vector3d(py::array_t<double> np_array);
double launch_simulation(unsigned long seed, int N, size_t K, double I, double R,
                         py::array_t<double> centers_np, py::array_t<double> normals_np,
                         py::dict all_samples);
