# pragma once

#include <Constants.hpp>
#include <Eigen/Eigen>
#include <math.h>
#include <cmath>

using namespace std;
using namespace Eigen;

struct Revelator {
    double R; // Revelator radius

    // Static variables for precomputation
    static inline double mollifier_area;
    static inline bool precomputed = false;

    static void precomputeArea() {
        // Precomputes area of mollifier (uses radial function)
        if(precomputed) return;
        const int N = 500;
        int numPoints = 2 * N + 1;
        double dr = R / N; // Integration step
        VectorXd rVec(numPoints);
        for (int i = 0; i < numPoints; ++i)
            rVec(i) = i * dr;
        ArrayXd weights(numPoints);
        weights(0) = 1.0;
        weights(numPoints - 1) = 1.0;
        for (int i = 1; i < numPoints - 1; ++i)
            // Assign 2 if i is even, else 4
            if (i % 2 == 0) weights(i) = 2; else weights(i) = 4;

        ArrayXd y(numPoints); // Values of function
        double r;
        for (size_t i = 0; i < rVec.length(); ++i) {
            r = rVec(i);
            y(i) = r * r * exp(1 / (1 - (r / R) * (r / R)));
        }
        mollifier_area = 4 * PI * (weights * y).sum();
        precomputed = true;
    }

    Revelator(double R){
        R = R;
    }

    double mollifier(Vector3d& X){
        r = X.norm() / R;
        return exp(1 / (1 - r * r));
    }

    double revelatorProbability(Vector3d& X) {
        precomputeArea(); // Precomputed mollifier Area
        return mollifier(X)/area;
    }
};
