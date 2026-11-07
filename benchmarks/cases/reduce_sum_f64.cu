// SOTA baseline for reduce_sum_f64: OUTPUT[0] = sum of INPUT[0..N).
// CUB's BlockReduce is the canonical reduction and its warp-shuffle path has no per-level barrier,
// so this kernel is the sharpest available contrast with the shared-tree fold the Forth side can
// express. One block, matching the case's single-block output, and each thread keeps a register
// accumulator fed by 128-bit loads.

#include <cub/block/block_reduce.cuh>

constexpr int kBlock = 256;
constexpr int kWidth = 2;

using BlockReduce = cub::BlockReduce<double, kBlock>;

extern "C" __global__ void reduce_sum_f64(long long n, const double *__restrict__ input,
                                          double *__restrict__ output) {
  __shared__ typename BlockReduce::TempStorage temp;

  const long long stride = static_cast<long long>(blockDim.x) * kWidth;
  long long i = static_cast<long long>(threadIdx.x) * kWidth;

  double sum = 0.0;
  for (; i + kWidth - 1 < n; i += stride) {
    const double2 values = *reinterpret_cast<const double2 *>(input + i);
    sum += values.x + values.y;
  }
  if (i < n) {
    sum += input[i];
  }

  sum = BlockReduce(temp).Sum(sum);
  if (threadIdx.x == 0) {
    output[0] = sum;
  }
}
