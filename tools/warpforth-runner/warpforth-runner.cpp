/// warpforth-runner: Execute PTX kernels via the CUDA Driver API.
///
/// Single-file C++ program designed to be uploaded and compiled on a remote
/// GPU host with `nvcc -o warpforth-runner warpforth-runner.cpp -lcuda
/// -std=c++17`.
///
/// Usage:
///   warpforth-runner kernel.ptx --param i32[]:1,2,3 --param f32:3.14 \
///       --grid 4,1,1 --block 64,1,1 --kernel main \
///       --output-param 0 --output-count 3

#include <cuda.h>

#include <charconv>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <string_view>
#include <system_error>
#include <type_traits>
#include <variant>
#include <vector>

#define CHECK_CU(call)                                                         \
  do {                                                                         \
    CUresult err = (call);                                                     \
    if (err != CUDA_SUCCESS) {                                                 \
      const char *errStr = nullptr;                                            \
      cuGetErrorString(err, &errStr);                                          \
      std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ << ": "     \
                << (errStr ? errStr : "unknown") << "\n";                      \
      exit(1);                                                                 \
    }                                                                          \
  } while (0)

template <typename T> struct ArrayParam {
  std::vector<T> values;
  CUdeviceptr devicePtr = 0;
};

template <typename T> struct ScalarParam {
  T value;
};

using Param = std::variant<ArrayParam<int32_t>, ArrayParam<float>,
                           ScalarParam<int32_t>, ScalarParam<float>>;

template <typename T> static void allocDevice(ArrayParam<T> &arr) {
  size_t bytes = arr.values.size() * sizeof(T);
  CHECK_CU(cuMemAlloc(&arr.devicePtr, bytes));
  CHECK_CU(cuMemcpyHtoD(arr.devicePtr, arr.values.data(), bytes));
}

template <typename T>
static void printOutput(ArrayParam<T> &arr, size_t count) {
  std::vector<T> output(arr.values.size());
  CHECK_CU(cuMemcpyDtoH(output.data(), arr.devicePtr,
                        arr.values.size() * sizeof(T)));
  for (size_t i = 0; i < count; ++i) {
    if (i > 0)
      std::cout << ",";
    if constexpr (std::is_floating_point_v<T>)
      std::cout << std::setprecision(9) << output[i];
    else
      std::cout << output[i];
  }
  std::cout << "\n";
}

static void *kernelArgPtr(Param &p) {
  if (auto *a = std::get_if<ArrayParam<int32_t>>(&p))
    return &a->devicePtr;
  if (auto *a = std::get_if<ArrayParam<float>>(&p))
    return &a->devicePtr;
  if (auto *s = std::get_if<ScalarParam<int32_t>>(&p))
    return &s->value;
  return &std::get<ScalarParam<float>>(p).value;
}

static bool isScalar(const Param &p) {
  return std::holds_alternative<ScalarParam<int32_t>>(p) ||
         std::holds_alternative<ScalarParam<float>>(p);
}

struct Dims {
  unsigned x = 1, y = 1, z = 1;
};

static int parseIntArg(std::string_view s, std::string_view optName) {
  int value = 0;
  auto [ptr, ec] = std::from_chars(s.data(), s.data() + s.size(), value);
  if (ec != std::errc{} || ptr != s.data() + s.size()) {
    std::cerr << "Error: " << optName << " expects an integer, got: " << s
              << "\n";
    exit(1);
  }
  return value;
}

static Dims parseDims(std::string_view s) {
  Dims d;
  const char *p = s.data();
  const char *end = s.data() + s.size();

  auto dimsErr = [&]() {
    std::cerr << "Error: expected 3 comma-separated values, got: " << s << "\n";
    exit(1);
  };

  auto [p1, ec1] = std::from_chars(p, end, d.x);
  if (ec1 != std::errc{} || p1 == end || *p1 != ',')
    dimsErr();

  auto [p2, ec2] = std::from_chars(p1 + 1, end, d.y);
  if (ec2 != std::errc{} || p2 == end || *p2 != ',')
    dimsErr();

  auto [p3, ec3] = std::from_chars(p2 + 1, end, d.z);
  if (ec3 != std::errc{} || p3 != end)
    dimsErr();

  return d;
}

static Param parseParam(std::string_view s) {
  std::string input(s);

  auto colonPos = input.find(':');
  if (colonPos == std::string::npos) {
    std::cerr << "Error: --param requires type prefix (e.g. i32:42 or "
                 "f32[]:1.0,2.0), got: "
              << s << "\n";
    exit(1);
  }

  std::string typePrefix = input.substr(0, colonPos);
  std::string valueStr = input.substr(colonPos + 1);

  if (valueStr.empty()) {
    std::cerr << "Error: --param requires at least one value, got: " << s
              << "\n";
    exit(1);
  }

  // Parse comma-separated values into a typed vector
  auto parseValues = [&](auto convert) {
    using T = decltype(convert(std::string{}));
    std::vector<T> vals;
    std::istringstream iss(valueStr);
    std::string token;
    while (std::getline(iss, token, ','))
      vals.push_back(convert(token));
    return vals;
  };

  auto toI32 = [&](const std::string &tok) -> int32_t {
    try {
      return std::stoi(tok);
    } catch (const std::exception &) {
      std::cerr << "Error: invalid integer value '" << tok << "' in --param "
                << s << "\n";
      exit(1);
    }
  };
  auto toF32 = [&](const std::string &tok) -> float {
    try {
      return std::stof(tok);
    } catch (const std::exception &) {
      std::cerr << "Error: invalid float value '" << tok << "' in --param " << s
                << "\n";
      exit(1);
    }
  };

  if (typePrefix == "i32[]")
    return Param{ArrayParam<int32_t>{parseValues(toI32)}};
  if (typePrefix == "f32[]")
    return Param{ArrayParam<float>{parseValues(toF32)}};

  // Scalars — must be exactly one value
  if (valueStr.find(',') != std::string::npos) {
    std::cerr << "Error: scalar param expects exactly one value, got: " << s
              << "\n";
    exit(1);
  }

  if (typePrefix == "i32")
    return Param{ScalarParam<int32_t>{toI32(valueStr)}};
  if (typePrefix == "f32")
    return Param{ScalarParam<float>{toF32(valueStr)}};

  std::cerr << "Error: unsupported param type '" << typePrefix
            << "' (expected i32, i32[], f32, or f32[]), got: " << s << "\n";
  exit(1);
}

static std::string readFile(std::string_view path) {
  std::ifstream f(std::string(path), std::ios::binary);
  if (!f) {
    std::cerr << "Error: cannot open " << path << "\n";
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
    std::string_view arg = argv[i];
    auto needsValue = [&](std::string_view opt) {
      if (++i >= argc) {
        std::cerr << "Error: " << opt << " requires a value\n";
        exit(1);
      }
    };
    if (arg == "--param") {
      needsValue("--param");
      params.push_back(parseParam(argv[i]));
    } else if (arg == "--grid") {
      needsValue("--grid");
      grid = parseDims(argv[i]);
    } else if (arg == "--block") {
      needsValue("--block");
      block = parseDims(argv[i]);
    } else if (arg == "--output-param") {
      needsValue("--output-param");
      outputParam = parseIntArg(argv[i], "--output-param");
    } else if (arg == "--output-count") {
      needsValue("--output-count");
      outputCount = parseIntArg(argv[i], "--output-count");
    } else if (arg == "--kernel") {
      needsValue("--kernel");
      kernelName = argv[i];
    } else if (arg[0] == '-') {
      std::cerr << "Error: unknown option " << arg << "\n";
      exit(1);
    } else {
      ptxFile = argv[i];
    }
  }

  if (!ptxFile) {
    std::cerr << "Usage: warpforth-runner kernel.ptx --kernel NAME "
                 "[--param i32[]:V,...] [--param f32[]:V,...] "
                 "[--param i32:V] [--param f32:V] [--grid X,Y,Z] "
                 "[--block X,Y,Z] [--output-param N] [--output-count N]\n";
    return 1;
  }

  if (!kernelName) {
    std::cerr << "Error: --kernel NAME is required\n";
    return 1;
  }

  if (params.empty()) {
    std::cerr << "Error: at least one --param is required\n";
    return 1;
  }

  if (outputParam < 0 || outputParam >= static_cast<int>(params.size())) {
    std::cerr << "Error: output-param " << outputParam << " out of range (have "
              << params.size() << " params)\n";
    return 1;
  }

  if (isScalar(params[outputParam])) {
    std::cerr << "Error: output-param " << outputParam
              << " is a scalar (cannot read back)\n";
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

  // Allocate device buffers for array params
  for (auto &p : params) {
    if (auto *a = std::get_if<ArrayParam<int32_t>>(&p))
      allocDevice(*a);
    else if (auto *a = std::get_if<ArrayParam<float>>(&p))
      allocDevice(*a);
  }

  // Set up kernel parameters — Driver API expects array of pointers to args
  std::vector<void *> kernelArgs(params.size());
  for (size_t i = 0; i < params.size(); ++i)
    kernelArgs[i] = kernelArgPtr(params[i]);

  // Launch kernel
  CHECK_CU(cuLaunchKernel(func, grid.x, grid.y, grid.z, block.x, block.y,
                          block.z, 0, nullptr, kernelArgs.data(), nullptr));

  CHECK_CU(cuCtxSynchronize());

  // Copy back and print output param
  size_t count = outputCount >= 0 ? static_cast<size_t>(outputCount) : 0;
  if (auto *iArr = std::get_if<ArrayParam<int32_t>>(&params[outputParam])) {
    if (outputCount < 0)
      count = iArr->values.size();
    printOutput(*iArr, count);
  } else {
    auto &fArr = std::get<ArrayParam<float>>(params[outputParam]);
    if (outputCount < 0)
      count = fArr.values.size();
    printOutput(fArr, count);
  }

  // Cleanup — only free device memory for array params
  for (auto &p : params) {
    if (auto *a = std::get_if<ArrayParam<int32_t>>(&p))
      cuMemFree(a->devicePtr);
    else if (auto *a = std::get_if<ArrayParam<float>>(&p))
      cuMemFree(a->devicePtr);
  }
  cuModuleUnload(module);
  cuCtxDestroy(ctx);

  return 0;
}
