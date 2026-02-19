/// warpforth-runner: Execute PTX kernels via the CUDA Driver API.
///
/// Single-file C++ program designed to be uploaded and compiled on a remote
/// GPU host with `nvcc -o warpforth-runner warpforth-runner.cpp -lcuda
/// -std=c++17`.
///
/// Usage:
///   warpforth-runner kernel.ptx --param i64[]:1,2,3 --param i64:42 \
///       --grid 4,1,1 --block 64,1,1 --kernel main \
///       --output-param 0 --output-count 3

#include <cuda.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#define CHECK_CU(call)                                                         \
  do {                                                                         \
    CUresult err = (call);                                                     \
    if (err != CUDA_SUCCESS) {                                                 \
      const char *errStr = nullptr;                                            \
      cuGetErrorString(err, &errStr);                                          \
      fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,         \
              errStr ? errStr : "unknown");                                    \
      exit(1);                                                                 \
    }                                                                          \
  } while (0)

enum class ParamKind { Array, Scalar };

struct Param {
  ParamKind kind = ParamKind::Array;
  std::vector<int64_t> values;
};

struct Dims {
  unsigned x = 1, y = 1, z = 1;
};

static Dims parseDims(const char *s) {
  Dims d;
  if (sscanf(s, "%u,%u,%u", &d.x, &d.y, &d.z) != 3) {
    fprintf(stderr, "Error: expected 3 comma-separated values, got: %s\n", s);
    exit(1);
  }
  return d;
}

static Param parseParam(const char *s) {
  Param p;
  std::string input(s);

  // Find the type prefix (everything before the first ':')
  auto colonPos = input.find(':');
  if (colonPos == std::string::npos) {
    fprintf(stderr,
            "Error: --param requires type prefix (e.g. i64:42 or "
            "i64[]:1,2,3), got: %s\n",
            s);
    exit(1);
  }

  std::string typePrefix = input.substr(0, colonPos);
  std::string valueStr = input.substr(colonPos + 1);

  if (valueStr.empty()) {
    fprintf(stderr, "Error: --param requires at least one value, got: %s\n", s);
    exit(1);
  }

  // Determine kind from type prefix
  if (typePrefix == "i64[]") {
    p.kind = ParamKind::Array;
  } else if (typePrefix == "i64") {
    p.kind = ParamKind::Scalar;
  } else {
    fprintf(stderr,
            "Error: unsupported param type '%s' (expected i64 or "
            "i64[]), got: %s\n",
            typePrefix.c_str(), s);
    exit(1);
  }

  // Parse values
  std::istringstream iss(valueStr);
  std::string token;
  while (std::getline(iss, token, ',')) {
    p.values.push_back(std::stoll(token));
  }

  if (p.kind == ParamKind::Scalar && p.values.size() != 1) {
    fprintf(stderr, "Error: scalar param expects exactly one value, got: %s\n",
            s);
    exit(1);
  }

  return p;
}

static std::string readFile(const char *path) {
  std::ifstream f(path, std::ios::binary);
  if (!f) {
    fprintf(stderr, "Error: cannot open %s\n", path);
    exit(1);
  }
  std::ostringstream ss;
  ss << f.rdbuf();
  return ss.str();
}

int main(int argc, char **argv) {
  const char *ptxFile = nullptr;
  const char *kernelName = nullptr;
  std::vector<Param> params;
  Dims grid, block;
  int outputParam = 0;
  int outputCount = -1; // -1 = all

  // Parse arguments
  for (int i = 1; i < argc; ++i) {
    if (strcmp(argv[i], "--param") == 0) {
      if (++i >= argc) {
        fprintf(stderr, "Error: --param requires a value\n");
        return 1;
      }
      params.push_back(parseParam(argv[i]));
    } else if (strcmp(argv[i], "--grid") == 0) {
      if (++i >= argc) {
        fprintf(stderr, "Error: --grid requires a value\n");
        return 1;
      }
      grid = parseDims(argv[i]);
    } else if (strcmp(argv[i], "--block") == 0) {
      if (++i >= argc) {
        fprintf(stderr, "Error: --block requires a value\n");
        return 1;
      }
      block = parseDims(argv[i]);
    } else if (strcmp(argv[i], "--output-param") == 0) {
      if (++i >= argc) {
        fprintf(stderr, "Error: --output-param requires a value\n");
        return 1;
      }
      outputParam = atoi(argv[i]);
    } else if (strcmp(argv[i], "--output-count") == 0) {
      if (++i >= argc) {
        fprintf(stderr, "Error: --output-count requires a value\n");
        return 1;
      }
      outputCount = atoi(argv[i]);
    } else if (strcmp(argv[i], "--kernel") == 0) {
      if (++i >= argc) {
        fprintf(stderr, "Error: --kernel requires a value\n");
        return 1;
      }
      kernelName = argv[i];
    } else if (argv[i][0] == '-') {
      fprintf(stderr, "Error: unknown option %s\n", argv[i]);
      return 1;
    } else {
      ptxFile = argv[i];
    }
  }

  if (!ptxFile) {
    fprintf(stderr, "Usage: warpforth-runner kernel.ptx --kernel NAME "
                    "[--param i64[]:V,...] [--param i64:V] [--grid X,Y,Z] "
                    "[--block X,Y,Z] [--output-param N] [--output-count N]\n");
    return 1;
  }

  if (!kernelName) {
    fprintf(stderr, "Error: --kernel NAME is required\n");
    return 1;
  }

  if (params.empty()) {
    fprintf(stderr, "Error: at least one --param is required\n");
    return 1;
  }

  if (outputParam < 0 || outputParam >= static_cast<int>(params.size())) {
    fprintf(stderr, "Error: output-param %d out of range (have %zu params)\n",
            outputParam, params.size());
    return 1;
  }

  if (params[outputParam].kind == ParamKind::Scalar) {
    fprintf(stderr, "Error: output-param %d is a scalar (cannot read back)\n",
            outputParam);
    return 1;
  }

  // Read PTX
  std::string ptx = readFile(ptxFile);

  // Initialize CUDA
  CHECK_CU(cuInit(0));

  CUdevice device;
  CHECK_CU(cuDeviceGet(&device, 0));

  CUcontext ctx;
  CHECK_CU(cuCtxCreate(&ctx, 0, device));

  // Load PTX module
  CUmodule module;
  CHECK_CU(cuModuleLoadData(&module, ptx.c_str()));

  CUfunction func;
  CHECK_CU(cuModuleGetFunction(&func, module, kernelName));

  // Allocate device buffers (arrays) or store scalar values
  std::vector<CUdeviceptr> devicePtrs(params.size(), 0);
  std::vector<int64_t> scalarValues(params.size(), 0);
  for (size_t i = 0; i < params.size(); ++i) {
    if (params[i].kind == ParamKind::Array) {
      size_t bytes = params[i].values.size() * sizeof(int64_t);
      CHECK_CU(cuMemAlloc(&devicePtrs[i], bytes));
      CHECK_CU(cuMemcpyHtoD(devicePtrs[i], params[i].values.data(), bytes));
    } else {
      scalarValues[i] = params[i].values[0];
    }
  }

  // Set up kernel parameters — Driver API expects array of pointers to args
  std::vector<void *> kernelArgs(params.size());
  for (size_t i = 0; i < params.size(); ++i) {
    kernelArgs[i] = (params[i].kind == ParamKind::Array)
                        ? static_cast<void *>(&devicePtrs[i])
                        : static_cast<void *>(&scalarValues[i]);
  }

  // Launch kernel
  CHECK_CU(cuLaunchKernel(func, grid.x, grid.y, grid.z, block.x, block.y,
                          block.z, 0, nullptr, kernelArgs.data(), nullptr));

  CHECK_CU(cuCtxSynchronize());

  // Copy back output param
  size_t outSize = params[outputParam].values.size();
  std::vector<int64_t> output(outSize);
  CHECK_CU(cuMemcpyDtoH(output.data(), devicePtrs[outputParam],
                        outSize * sizeof(int64_t)));

  // Print CSV to stdout
  size_t count = outputCount >= 0 ? static_cast<size_t>(outputCount) : outSize;
  for (size_t i = 0; i < count; ++i) {
    if (i > 0)
      printf(",");
    printf("%ld", static_cast<long>(output[i]));
  }
  printf("\n");

  // Cleanup — only free device memory for array params
  for (size_t i = 0; i < params.size(); ++i) {
    if (params[i].kind == ParamKind::Array)
      cuMemFree(devicePtrs[i]);
  }
  cuModuleUnload(module);
  cuCtxDestroy(ctx);

  return 0;
}
