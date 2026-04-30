#include "Simulation.hpp"
#include "Utils.hpp"
#include <BField.hpp>
#include <Revelator.hpp>
#include <Eigen/Eigen>
#include <vector>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>
#include <iostream>
#include <chrono>
#include <unordered_map>
#include <string>

using namespace std;
using namespace Eigen;
namespace py = pybind11;

std::vector<Eigen::Vector3d> numpy_to_vector3d(py::array_t<double> np_array)
{
    // Convert numpy to Eigen
    auto r = np_array.unchecked<2>();
    std::vector<Eigen::Vector3d> vector_eigen;
    vector_eigen.reserve(r.shape(0));
    for (int i = 0; i < r.shape(0); i++) {
        vector_eigen.emplace_back(r(i, 0), r(i, 1), r(i, 2));
    }
    return vector_eigen;
}

std::vector<double> numpy_to_vector(py::array_t<double> np_array)
{
    // Copy a NumPy array to std::vector
    auto r = np_array.unchecked<1>();
    std::vector<double> vector_cpp;
    vector_cpp.reserve(r.shape(0));
    for (ssize_t i = 0; i < r.shape(0); i++) {
        vector_cpp.push_back(r(i));
    }
    return vector_cpp;
}

std::unordered_map<std::string, std::vector<double>> dict_to_map(py::dict py_dict)
{
    // Copy a NumPy dict to std::unordered_map
    std::unordered_map<string, vector<double>> map_cpp;
    for (auto item : py_dict) {
        std::string key = py::cast<std::string>(item.first);
        // Cast the value to a numpy array
        std::vector<double> val = numpy_to_vector(py::cast<py::array_t<double>>(item.second));
        map_cpp[key] = std::move(val);
    }
    return map_cpp;
}

double launch_simulation(unsigned long seed, int N, size_t K, double I, double R,
                         py::array_t<double> centers_np, py::array_t<double> normals_np,
                         py::dict samples_np)
{
    // Run main simulation
    chrono::steady_clock::time_point t_begin = chrono::steady_clock::now();
    // Simulation specifics
    const double T = 1e9; // K
    double dt = 1e-9; // Initial time step

    // Define Revelator
    double rho = 2;
    Revelator revelator(rho);

    // Define expected dose
    double totalExpectedDose;

    // Convert numpy arrays
    vector<Vector3d> centers = numpy_to_vector3d(centers_np);
    vector<Vector3d> normals = numpy_to_vector3d(normals_np);

    // Convert the samples dictionary
    unordered_map<string, vector<double>> samples = dict_to_map(samples_np);

    // Try to run simulation
    try {
        // Build magnetic field
        BField field(K, centers, normals, R, I);

        // Run multithread simulation from CSV input (for particles phyisics data)
        totalExpectedDose = runFromCSV_MT("../data/particles_input.csv", field, revelator, samples, N, dt, seed);

        chrono::steady_clock::time_point t_end = chrono::steady_clock::now();
        double elapsedTime = chrono::duration_cast<chrono::seconds>(t_end-t_begin).count();
        cout << "Elapsed simulation time: " << elapsedTime << "s." << endl;
        cout << "Total expected dose: " << totalExpectedDose << endl;
    } catch (const invalid_argument e){
        cerr << "Error: number of coils and number of coils parameters provided do not match!" << endl;
    }

    return totalExpectedDose;
}
