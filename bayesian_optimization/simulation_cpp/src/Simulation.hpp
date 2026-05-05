#pragma once

#include <Eigen/Eigen>
#include <vector>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <utility>

using namespace std;
using namespace Eigen;
namespace py = pybind11;

std::vector<Eigen::Vector3d> numpy_to_vector3d(py::array_t<double> np_array);
std::vector<double> numpy_to_vector(py::array_t<double> np_array);
std::unordered_map<std::string, std::vector<double>> dict_to_map(py::dict py_dict);
pair<double, double> launch_simulation(unsigned long seed, int N, size_t K, double I, double R,
                         py::array_t<double> centers_np, py::array_t<double> normals_np,
                         py::dict all_samples);
