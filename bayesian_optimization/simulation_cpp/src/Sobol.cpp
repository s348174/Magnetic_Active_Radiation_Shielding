#include <vector>
#include <cmath>
#include <stdexcept>
#include <Eigen/Dense>

using Vector3d = Eigen::Vector3d;
using namespace std;

// ---------------------------------------------------------------------------
// Sobol sequence generator (base-2 direction numbers, first 2 dimensions)
// This is a minimal, self-contained implementation.
// ---------------------------------------------------------------------------

namespace detail {

// Direction numbers for dimensions 1 and 2 (Joe & Kuo, 2010 table)
// Dimension 1 is always powers of 2 shifted right.
// Dimension 2 uses primitive polynomial x^2 + x + 1 (s=2, a=1)
static constexpr int MAX_BITS = 32;

inline void sobol_init(uint32_t V[2][MAX_BITS]) {
    // Dimension 1: V[0][i] = 2^(MAX_BITS-1-i)
    for (int i = 0; i < MAX_BITS; ++i)
        V[0][i] = 1u << (MAX_BITS - 1 - i);

    // Dimension 2: s=2, a=1, m = {1, 1}
    int s = 2;
    uint32_t m[2] = {1, 1};
    for (int i = 0; i < s; ++i)
        V[1][i] = m[i] << (MAX_BITS - 1 - i);
    for (int i = s; i < MAX_BITS; ++i) {
        uint32_t v = V[1][i - s] >> s;
        v ^= V[1][i - s];
        // a=1 has only bit 0 set between bits s-2..1
        // recurrence: V[i] = V[i-s] ^ (V[i-s] >> s) ^ V[i-1]
        // (for a=1 the XOR loop has just the one inner bit)
        v ^= V[1][i - 1];
        V[1][i] = v;
    }
}

// XOR-shift hash for Owen scrambling (fast, good avalanche)
static uint32_t owenHash(uint32_t x, uint32_t seed) {
    x ^= seed;
    x ^= x >> 17; x ^= x << 31; x ^= x >> 8;
    return x;
}

// Fill `n` 2D Sobol points into out[][2] as doubles in [0,1)
inline void sobol2d(int n, std::vector<std::array<double, 2>>& out, uint32_t seed) {
    uint32_t V[2][MAX_BITS];
    sobol_init(V);

    out.resize(n);
    uint32_t x[2] = {0, 0};
    for (int i = 0; i < n; ++i) {
        // Gray-code index trick: find rightmost zero bit of i
        int c = 0;
        int val = i;
        while (val & 1) { ++c; val >>= 1; }
        x[0] ^= V[0][c];
        x[1] ^= V[1][c];
        out[i][0] = owenHash(x[0], seed)        * (1.0 / (1ull << MAX_BITS));
        out[i][1] = owenHash(x[1], seed * 2654435761u) * (1.0 / (1ull << MAX_BITS));
        out[i][0] = x[0] * (1.0 / (1ull << MAX_BITS));
        out[i][1] = x[1] * (1.0 / (1ull << MAX_BITS));
    }
}

} // namespace detail

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Sample N points uniformly on the surface of a sphere of radius R
/// using a 2D Sobol low-discrepancy sequence mapped via equal-area projection.
///
/// @param N  Number of sample points (> 0)
/// @param R  Sphere radius (> 0)
/// @return   Vector of N cartesian points on the sphere surface
vector<Vector3d> sampleSphereSobol(int N, double R, uint32_t seed) {
    if (N <= 0) throw invalid_argument("N must be positive");
    if (R <= 0.0) throw invalid_argument("R must be positive");

    // Generate 2D Sobol samples in [0,1)^2
    vector<array<double, 2>> uv;
    detail::sobol2d(N, uv, seed);

    vector<Vector3d> points;
    points.reserve(N);

    for (int i = 0; i < N; ++i) {
        // Equal-area (Shirley & Chiu / Archimedes) mapping:
        //   u -> azimuthal angle phi in [0, 2*pi)
        //   v -> z = 1 - 2v  (uniform in [-1, 1])
        // This gives a perfectly uniform distribution on the sphere.
        const double phi = 2.0 * M_PI * uv[i][0];
        const double z   = 1.0 - 2.0 * uv[i][1];          // in [-1, 1]
        const double r   = sqrt(max(0.0, 1.0 - z * z)); // sin(theta)

        points.emplace_back(R * r * cos(phi),
                            R * r * sin(phi),
                            R * z);
    }
    return points;
}
