#include <pybind11/pybind11.h>
#include "Simulation.hpp"

namespace py = pybind11;

PYBIND11_MODULE(simulator, m) {
    m.doc() = "Minimal test simulator module";

    // Expose dummy_simulation to Python
    m.def("dummy_simulation", &dummy_simulation, "A dummy simulation function");
}
