// SOTA baseline for axpy_f64: Y[i] = ALPHA * X[i] + Y[i].
// The case is bandwidth bound, so a library cannot beat a hand-written kernel that already moves
// one 128-bit load pair per side per thread per turn. The fused multiply-add and the 128-bit loads
// are both beyond the current Forth word set.

#include <cmath>

constexpr int kWidth = 2;

extern "C" __global__ void axpy_f64(long long n, double alpha,
                                    const double *__restrict__ x,
                                    double *__restrict__ y) {
  const long long stride = static_cast<long long>(gridDim.x) * blockDim.x;
  const long long gid = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;

  long long i = gid * kWidth;
  for (; i + kWidth - 1 < n; i += stride * kWidth) {
    const double2 xv = *reinterpret_cast<const double2 *>(x + i);
    const double2 yv = *reinterpret_cast<const double2 *>(y + i);
    const double2 out = {fma(alpha, xv.x, yv.x), fma(alpha, xv.y, yv.y)};
    *reinterpret_cast<double2 *>(y + i) = out;
  }
  if (i < n) {
    y[i] = fma(alpha, x[i], y[i]);
  }
}
