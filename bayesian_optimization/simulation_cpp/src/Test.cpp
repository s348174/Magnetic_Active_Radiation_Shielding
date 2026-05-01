#include <Test.hpp>
#include <Utils.hpp>
#include <BField.hpp>
#include <Particle.hpp>
#include <Eigen/Eigen>
#include <array>
#include <chrono>
#include <iostream>

using namespace std;
using namespace Eigen;

void computeMetrics(vector<Vector3d> positions, vector<Vector3d> speeds)
{
    double m = amu;
    double q = e_q;
    double dt = 1e-9; // Initial time step
    double T_max = 100 / 1e6;
    if (!(positions.size() == speeds.size())) {
        cerr << "Input size not matching. Abort" << endl;
        array<double, 2> e = {0,0};
        return;
    }
    double k = 1;
    Vector3d center;
    center << 0, 0, 0;
    vector<Vector3d> centers;
    centers.push_back(center);
    Vector3d normal;
    normal << 0, 0, 1;
    vector<Vector3d> normals;
    normals.push_back(normal);
    double I = 1e5;
    double R = 5;
    BField field(k, centers, normals, I, R);
    Revelator detector(R);

    chrono::steady_clock::time_point bs_begin = chrono::steady_clock::now();
    vector<Vector3d> baseline;
    for (size_t i = 0; i < positions.size(); ++i) {
        Vector3d X0 = positions[i];
        Vector3d v0 = speeds[i];
        Particle part(m ,q, X0, v0, T_max, dt);

        // Start trajectory computation
        double t = 0;
        while (t < T_max) {
            part.updatePositionRK4_BS(field, detector);
            t += part.dt;
        }
        baseline.push_back(part.X_t);
    }
    chrono::steady_clock::time_point bs_end = chrono::steady_clock::now();
    double elapsedTime_base = chrono::duration_cast<chrono::milliseconds>(bs_end-bs_begin).count();

    chrono::steady_clock::time_point f_begin = chrono::steady_clock::now();
    vector<Vector3d> fast;
    for (size_t i = 0; i < positions.size(); ++i) {
        Vector3d X0 = positions[i];
        Vector3d v0 = speeds[i];
        Particle part(m ,q, X0, v0, T_max, dt);

        // Start trajectory computation
        double t = 0;
        while (t < T_max) {
            part.updatePositionRK4_BS(field, detector);
            t += part.dt;
        }
        fast.push_back(part.X_t);
    }
    chrono::steady_clock::time_point f_end = chrono::steady_clock::now();
    double elapsedTime_fast = chrono::duration_cast<chrono::milliseconds>(f_end-f_begin).count();

    double cumErr = 0;
    double relErr = 0;
    for (size_t i = 0; i < fast.size(); ++i) {
        Vector3d diff = fast[i] - baseline[i];
        cumErr += diff.norm();
        relErr += diff.norm() / baseline[i].norm();
    }

    chrono::steady_clock::time_point boris_begin = chrono::steady_clock::now();
    vector<Vector3d> boris;
    for (size_t i = 0; i < positions.size(); ++i) {
        Vector3d X0 = positions[i];
        Vector3d v0 = speeds[i];
        Particle part(m ,q, X0, v0, T_max, dt);

        // Start trajectory computation
        double t = 0;
        while (t < T_max) {
            part.updateBS(field, detector);
            t += part.dt;
        }
        boris.push_back(part.X_t);
    }
    chrono::steady_clock::time_point boris_end = chrono::steady_clock::now();
    double elapsedTime_boris = chrono::duration_cast<chrono::milliseconds>(boris_end-boris_begin).count();

    double borisErr = 0;
    double borisRel = 0;
    for (size_t i = 0; i < boris.size(); ++i) {
        Vector3d diff = boris[i] - baseline[i];
        borisErr += diff.norm();
        borisRel += diff.norm() / baseline[i].norm();
    }

    chrono::steady_clock::time_point sam_begin = chrono::steady_clock::now();
    vector<Vector3d> sam;
    for (size_t i = 0; i < positions.size(); ++i) {
        Vector3d X0 = positions[i];
        Vector3d v0 = speeds[i];
        Particle part(m ,q, X0, v0, T_max, dt);

        // Start trajectory computation
        double t = 0;
        while (t < T_max) {
            part.updatePositionRK4(field, detector);
            t += part.dt;
        }
        sam.push_back(part.X_t);
    }
    chrono::steady_clock::time_point sam_end = chrono::steady_clock::now();
    double elapsedTime_sam = chrono::duration_cast<chrono::milliseconds>(sam_end-sam_begin).count();

    double samErr = 0;
    double samRel = 0;
    for (size_t i = 0; i < sam.size(); ++i) {
        Vector3d diff = sam[i] - baseline[i];
        samErr += diff.norm();
        samRel += diff.norm() / baseline[i].norm();
    }

    cout << "Baseline (RK4-BS) Time: " << elapsedTime_base << " ms" << endl;
    cout << "Fast (Boris-SAM) Time: " << elapsedTime_fast << " ms" << endl;
    cout << "Boris-BS Time: " << elapsedTime_boris << " ms" << endl;
    cout << "RK4-SAM Time: " << elapsedTime_sam << " ms" << endl;
    cout << "Baseline vs Fast: Cumulative Error = " << cumErr << ", Relative Error = " << relErr << endl;
    cout << "Baseline vs Boris: Cumulative Error = " << borisErr << ", Relative Error = " << borisRel << endl;
    cout << "Baseline vs SAM: Cumulative Error = " << samErr << ", Relative Error = " << samRel << endl;
}