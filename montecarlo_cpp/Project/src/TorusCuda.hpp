#pragma once

#include <array>

struct Torus;

namespace mars {

// Computes B(x,y,z) for the torus using CUDA-parallel Simpson integration.
// Returns true on success, false on CUDA/runtime failure.
bool computeTorusMagneticFieldCUDA(const Torus& torus,
                                   double x,
                                   double y,
                                   double z,
                                   std::array<double, 3>& bOut);

} // namespace mars
