// SOTA baseline for scale_i64: OUT[i] = IN[i] * SCALE.
// No canonical library covers i64 scaling, so this is hand-written. It reads and writes 128 bits
// per thread per turn, which halves the instruction count and reaches the memory-bandwidth ceiling
// for the traffic the case moves.

constexpr int kPair = 2;

extern "C" __global__ void scale_i64(long long n, long long scale,
                                     const long long *__restrict__ in,
                                     long long *__restrict__ out) {
  const long long stride = static_cast<long long>(gridDim.x) * blockDim.x;
  const long long gid = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;

  long long i = gid * kPair;
  for (; i + kPair - 1 < n; i += stride * kPair) {
    const longlong2 packed = *reinterpret_cast<const longlong2 *>(in + i);
    const longlong2 scaled = {packed.x * scale, packed.y * scale};
    *reinterpret_cast<longlong2 *>(out + i) = scaled;
  }
  if (i < n) {
    out[i] = in[i] * scale;
  }
}
