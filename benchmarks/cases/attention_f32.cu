// SOTA baseline for attention_f32: O = softmax(scale * Q K^T) V, f32, non-causal.
// One warp per query row, online softmax fused with the output accumulation in a single pass over
// K and V, so a row of scores never reaches memory. Warp shuffles carry both reductions and each
// lane holds its slice of the output in registers across the whole key loop.
// HDIM must be a multiple of 32 and at most 128, because a lane owns HDIM/32 of it in registers.

#include <cmath>

constexpr int kWarpsPerBlock = 4;
constexpr int kLane = 32;
constexpr int kMaxPerLane = 4;

__device__ __forceinline__ float warp_sum(float value) {
#pragma unroll
  for (int offset = kLane / 2; offset > 0; offset /= 2) {
    value += __shfl_xor_sync(0xffffffffu, value, offset);
  }
  return value;
}

extern "C" __global__ void attention_f32(long long nrow, long long nkey, long long hdim,
                                         double scale, const float *__restrict__ q,
                                         const float *__restrict__ k,
                                         const float *__restrict__ v,
                                         float *__restrict__ o) {
  const int lane = threadIdx.x & (kLane - 1);
  const long long warp = (long long)blockIdx.x * kWarpsPerBlock + (threadIdx.x / kLane);
  if (warp >= nrow) {
    return;
  }

  const int per_lane = (int)(hdim / kLane);
  const float qscale = (float)scale;

  float qreg[kMaxPerLane];
#pragma unroll
  for (int slot = 0; slot < kMaxPerLane; ++slot) {
    qreg[slot] = slot < per_lane ? q[warp * hdim + lane * per_lane + slot] * qscale : 0.0f;
  }

  float acc[kMaxPerLane];
#pragma unroll
  for (int slot = 0; slot < kMaxPerLane; ++slot) {
    acc[slot] = 0.0f;
  }
  float row_max = -INFINITY;
  float row_sum = 0.0f;

  for (long long key = 0; key < nkey; ++key) {
    const float *krow = k + key * hdim;
    const float *vrow = v + key * hdim;

    float dot = 0.0f;
#pragma unroll
    for (int slot = 0; slot < kMaxPerLane; ++slot) {
      if (slot < per_lane) {
        dot = fmaf(qreg[slot], krow[lane * per_lane + slot], dot);
      }
    }
    const float score = warp_sum(dot);

    const float new_max = fmaxf(row_max, score);
    const float correction = __expf(row_max - new_max);
    const float weight = __expf(score - new_max);
    row_sum = row_sum * correction + weight;

#pragma unroll
    for (int slot = 0; slot < kMaxPerLane; ++slot) {
      if (slot < per_lane) {
        acc[slot] = fmaf(weight, vrow[lane * per_lane + slot], acc[slot] * correction);
      }
    }
    row_max = new_max;
  }

  const float norm = 1.0f / row_sum;
#pragma unroll
  for (int slot = 0; slot < kMaxPerLane; ++slot) {
    if (slot < per_lane) {
      o[warp * hdim + lane * per_lane + slot] = acc[slot] * norm;
    }
  }
}
