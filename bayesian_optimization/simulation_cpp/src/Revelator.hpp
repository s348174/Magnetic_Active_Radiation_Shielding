# pragma once

#include <Constants.hpp>
#include <Eigen/Eigen>
#include <math.h>
#include <cmath>
#include <iostream>

using namespace std;
using namespace Eigen;

struct Revelator {
    double R; // Revelator radius
    double mollifier_area; // Area for normalization

    Revelator(double R_in){
        R = R_in;
        // Precompute mollifier area
        const int N = 1000;
        int numPoints = 2 * N + 1;
        double dr = R / numPoints; // Integration step
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
        double rho;
        for (size_t i = 0; i < rVec.size(); ++i) {
            rho = rVec(i) / R;
            if (rho < 1 - 1e-6)
                y(i) = exp(1 / (rho * rho - 1));
            else
                y(i) = 0;
        }
        mollifier_area = 2 * (dr / 3) * (weights * y).sum();
        cout << "Mollifier area: " << mollifier_area;
    }

    double revelatorProbability(Vector3d& X) {
        double rho = X.norm() / R;
        if (rho < 1 - 1e-6) return exp(1 / (rho * rho - 1)) / mollifier_area;
        else return 0;
    }
};
