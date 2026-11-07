#include <pybind11/pybind11.h>
#include <pybind11/stl.h>      // std::vector bindings
#include <pybind11/eigen.h>    // Eigen bindings
#include "Simulation.hpp"

namespace py = pybind11;

PYBIND11_MODULE(simulator, m) {
    m.doc() = "Simulator module launcher";

    // Expose launch_simulation to Python
    m.def("launch_simulation", &launch_simulation, "Launch simulation function",
          py::arg("seed"), py::arg("N"), py::arg("K"), py::arg("I"), py::arg("R"),
          py::arg("centers"), py::arg("normals"), py::arg("samples"),
          py::arg("strata") = py::dict());
}
