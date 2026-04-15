#include "TorusCuda.hpp"

#include "Constants.hpp"
#include "Torus.hpp"

#include <cuda_runtime.h>

#include <array>
#include <cmath>
#include <iostream>

namespace {

__device__ inline double simpsonWeight(const int i, const int numPoints) {
    if (i == 0 || i == numPoints - 1) {
        return 1.0;
    }
    return (i % 2 == 0) ? 2.0 : 4.0;
}

__global__ void torusFieldKernel(const double x,
                                 const double y,
                                 const double z,
                                 const double R,
                                 const int numPoints,
                                 const double dtheta,
                                 double* sumX,
                                 double* sumY,
                                 double* sumZ) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= numPoints) {
        return;
    }

    const double theta = static_cast<double>(i) * dtheta;
    const double sinT = sin(theta);
    const double cosT = cos(theta);

    const double base = x * x + y * y + z * z + R * R - 2.0 * R * x * cosT - 2.0 * R * y * sinT;
    const double denom = base * sqrt(base + 1e-30);
    const double w = simpsonWeight(i, numPoints);

    const double fx = z * cosT / denom;
    const double fy = z * sinT / denom;
    const double fz = (R - x * cosT - y * sinT) / denom;

    atomicAdd(sumX, w * fx);
    atomicAdd(sumY, w * fy);
    atomicAdd(sumZ, w * fz);
}

} // namespace

namespace mars {

bool computeTorusMagneticFieldCUDA(const Torus& torus,
                                   const double x,
                                   const double y,
                                   const double z,
                                   std::array<double, 3>& bOut) {
    bOut = {0.0, 0.0, 0.0};

    const int numPoints = Torus::numPoints;
    const double dtheta = Torus::dtheta;

    // Reuse device scalars per host thread to avoid malloc/free in the hot loop.
    thread_local double *dSumX = nullptr, *dSumY = nullptr, *dSumZ = nullptr;
    thread_local bool allocated = false;
    if (!allocated) {
        if (cudaMalloc(&dSumX, sizeof(double)) != cudaSuccess ||
            cudaMalloc(&dSumY, sizeof(double)) != cudaSuccess ||
            cudaMalloc(&dSumZ, sizeof(double)) != cudaSuccess) {
            if (dSumX) cudaFree(dSumX);
            if (dSumY) cudaFree(dSumY);
            if (dSumZ) cudaFree(dSumZ);
            dSumX = nullptr;
            dSumY = nullptr;
            dSumZ = nullptr;
            return false;
        }
        allocated = true;
    }

    cudaMemset(dSumX, 0, sizeof(double));
    cudaMemset(dSumY, 0, sizeof(double));
    cudaMemset(dSumZ, 0, sizeof(double));

    constexpr int blockSize = 256;
    const int gridSize = (numPoints + blockSize - 1) / blockSize;

    torusFieldKernel<<<gridSize, blockSize>>>(x, y, z, torus.R, numPoints, dtheta, dSumX, dSumY, dSumZ);

    const cudaError_t launchErr = cudaGetLastError();
    const cudaError_t syncErr = cudaDeviceSynchronize();
    if (launchErr != cudaSuccess || syncErr != cudaSuccess) {
        return false;
    }

    double sumX = 0.0;
    double sumY = 0.0;
    double sumZ = 0.0;

    const cudaError_t copyXErr = cudaMemcpy(&sumX, dSumX, sizeof(double), cudaMemcpyDeviceToHost);
    const cudaError_t copyYErr = cudaMemcpy(&sumY, dSumY, sizeof(double), cudaMemcpyDeviceToHost);
    const cudaError_t copyZErr = cudaMemcpy(&sumZ, dSumZ, sizeof(double), cudaMemcpyDeviceToHost);

    if (copyXErr != cudaSuccess || copyYErr != cudaSuccess || copyZErr != cudaSuccess) {
        return false;
    }

    const double coeff = mu0 * torus.I * torus.R / (4.0 * PI);
    const double factor = dtheta / 3.0;

    bOut[0] = coeff * sumX * factor;
    bOut[1] = coeff * sumY * factor;
    bOut[2] = coeff * sumZ * factor;
    return true;
}

} // namespace mars
