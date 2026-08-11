/// warpforth-runner: Execute PTX kernels via the CUDA Driver API.
///
/// Single-file C++ program designed to be uploaded and compiled on a remote
/// GPU host with `nvcc -o warpforth-runner warpforth-runner.cpp -lcuda
/// -std=c++17`.
///
/// Usage:
///   warpforth-runner kernel.ptx --param i64[]:1,2,3 --param f64:3.14 \
///       --grid 4,1,1 --block 64,1,1 --kernel main \
///       --output-param 0 --output-count 3

#ifndef WARPFORTH_RUNNER_HOST_VALIDATION
#include <cuda.h>
#endif

#include <cerrno>
#include <charconv>
#include <cinttypes>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>

template <typename T> struct ArrayParam {
  std::vector<T> values;
};

template <typename T> struct ScalarParam {
  T value;
};

using Param = std::variant<ArrayParam<int64_t>, ArrayParam<double>,
                           ScalarParam<int64_t>, ScalarParam<double>>;

static std::runtime_error paramError(std::string_view kind,
                                     std::string_view token,
                                     std::string_view param,
                                     std::string_view reason) {
  std::ostringstream message;
  message << "invalid " << kind << " value '" << token << "' in --param "
          << param << ": " << reason;
  return std::runtime_error(message.str());
}

static int64_t parseI64(std::string_view token, std::string_view param) {
  if (token.empty())
    throw paramError("integer", token, param, "empty value");

  std::string value(token);
  char *end = nullptr;
  errno = 0;
  intmax_t parsed = std::strtoimax(value.c_str(), &end, 10);
  if (end == value.c_str())
    throw paramError("integer", token, param, "expected an integer");
  if (errno == ERANGE || parsed < INT64_MIN || parsed > INT64_MAX)
    throw paramError("integer", token, param, "value is out of range for i64");
  if (end != value.c_str() + value.size())
    throw paramError("integer", token, param, "trailing characters");
  return static_cast<int64_t>(parsed);
}

static double parseF64(std::string_view token, std::string_view param) {
  if (token.empty())
    throw paramError("float", token, param, "empty value");

  std::string value(token);
  char *end = nullptr;
  errno = 0;
  double parsed = std::strtod(value.c_str(), &end);
  if (end == value.c_str())
    throw paramError("float", token, param, "expected a float");
  if (errno == ERANGE)
    throw paramError("float", token, param, "value is out of range for f64");
  if (end != value.c_str() + value.size())
    throw paramError("float", token, param, "trailing characters");
  return parsed;
}

template <typename T, typename Parse>
static std::vector<T> parseArrayValues(std::string_view values,
                                       std::string_view param, Parse parse) {
  std::vector<T> result;
  size_t start = 0;
  while (true) {
    size_t comma = values.find(',', start);
    std::string_view token = values.substr(start, comma - start);
    if (token.empty())
      throw paramError(std::is_same_v<T, int64_t> ? "integer" : "float", token,
                       param, "empty comma-separated field");
    result.push_back(parse(token, param));
    if (comma == std::string_view::npos)
      break;
    start = comma + 1;
  }
  return result;
}

static Param parseParam(std::string_view input) {
  size_t colon = input.find(':');
  if (colon == std::string_view::npos)
    throw std::runtime_error(
        "--param requires type prefix (e.g. i64:42 or f64[]:1.0,2.0), got: " +
        std::string(input));

  std::string_view type = input.substr(0, colon);
  std::string_view values = input.substr(colon + 1);
  if (values.empty())
    throw std::runtime_error("--param requires at least one value, got: " +
                             std::string(input));

  if (type == "i64[]")
    return ArrayParam<int64_t>{
        parseArrayValues<int64_t>(values, input, parseI64)};
  if (type == "f64[]")
    return ArrayParam<double>{
        parseArrayValues<double>(values, input, parseF64)};

  if (values.find(',') != std::string_view::npos)
    throw std::runtime_error("scalar param expects exactly one value, got: " +
                             std::string(input));
  if (type == "i64")
    return ScalarParam<int64_t>{parseI64(values, input)};
  if (type == "f64")
    return ScalarParam<double>{parseF64(values, input)};

  throw std::runtime_error(
      "unsupported param type '" + std::string(type) +
      "' (expected i64, i64[], f64, or f64[]), got: " + std::string(input));
}

struct Dims {
  unsigned x = 1, y = 1, z = 1;
};

static int parseIntArg(std::string_view input, std::string_view option) {
  int value = 0;
  auto [end, error] =
      std::from_chars(input.data(), input.data() + input.size(), value);
  if (error != std::errc{} || end != input.data() + input.size())
    throw std::runtime_error(std::string(option) +
                             " expects an integer, got: " + std::string(input));
  return value;
}

static Dims parseDims(std::string_view input) {
  Dims dims;
  const char *current = input.data();
  const char *end = input.data() + input.size();

  auto parseError = [&]() {
    throw std::runtime_error("expected 3 comma-separated values, got: " +
                             std::string(input));
  };

  auto [firstEnd, firstError] = std::from_chars(current, end, dims.x);
  if (firstError != std::errc{} || firstEnd == end || *firstEnd != ',')
    parseError();

  auto [secondEnd, secondError] = std::from_chars(firstEnd + 1, end, dims.y);
  if (secondError != std::errc{} || secondEnd == end || *secondEnd != ',')
    parseError();

  auto [thirdEnd, thirdError] = std::from_chars(secondEnd + 1, end, dims.z);
  if (thirdError != std::errc{} || thirdEnd != end)
    parseError();

  return dims;
}

static bool isScalar(const Param &param) {
  return std::holds_alternative<ScalarParam<int64_t>>(param) ||
         std::holds_alternative<ScalarParam<double>>(param);
}

static size_t arraySize(const Param &param) {
  if (auto *array = std::get_if<ArrayParam<int64_t>>(&param))
    return array->values.size();
  return std::get<ArrayParam<double>>(param).values.size();
}

#ifndef WARPFORTH_RUNNER_HOST_VALIDATION
static void checkCu(CUresult result, std::string_view operation) {
  if (result == CUDA_SUCCESS)
    return;
  const char *description = nullptr;
  cuGetErrorString(result, &description);
  throw std::runtime_error(std::string(operation) + " failed: " +
                           (description ? description : "unknown CUDA error"));
}

static void reportCleanupError(CUresult result, std::string_view operation) {
  if (result == CUDA_SUCCESS)
    return;
  const char *description = nullptr;
  cuGetErrorString(result, &description);
  std::cerr << "CUDA cleanup error: " << operation
            << " failed: " << (description ? description : "unknown CUDA error")
            << "\n";
}

class CudaContext {
public:
  static CudaContext create(CUdevice device) {
    CUcontext context = nullptr;
    checkCu(cuCtxCreate(&context, 0, device), "cuCtxCreate");
    return CudaContext(context);
  }

  CudaContext(const CudaContext &) = delete;
  CudaContext &operator=(const CudaContext &) = delete;
  CudaContext(CudaContext &&other) noexcept
      : context_(std::exchange(other.context_, nullptr)) {}
  CudaContext &operator=(CudaContext &&other) noexcept {
    if (this != &other) {
      reset();
      context_ = std::exchange(other.context_, nullptr);
    }
    return *this;
  }
  ~CudaContext() { reset(); }

private:
  explicit CudaContext(CUcontext context) : context_(context) {}
  void reset() noexcept {
    if (context_) {
      reportCleanupError(cuCtxDestroy(context_), "cuCtxDestroy");
      context_ = nullptr;
    }
  }

  CUcontext context_ = nullptr;
};

class CudaModule {
public:
  static CudaModule load(const char *ptx) {
    CUmodule module = nullptr;
    checkCu(cuModuleLoadData(&module, ptx), "cuModuleLoadData");
    return CudaModule(module);
  }

  CudaModule(const CudaModule &) = delete;
  CudaModule &operator=(const CudaModule &) = delete;
  CudaModule(CudaModule &&other) noexcept
      : module_(std::exchange(other.module_, nullptr)) {}
  CudaModule &operator=(CudaModule &&other) noexcept {
    if (this != &other) {
      reset();
      module_ = std::exchange(other.module_, nullptr);
    }
    return *this;
  }
  ~CudaModule() { reset(); }

  CUmodule get() const { return module_; }

private:
  explicit CudaModule(CUmodule module) : module_(module) {}
  void reset() noexcept {
    if (module_) {
      reportCleanupError(cuModuleUnload(module_), "cuModuleUnload");
      module_ = nullptr;
    }
  }

  CUmodule module_ = nullptr;
};

class DeviceAllocation {
public:
  DeviceAllocation(const void *source, size_t bytes) {
    checkCu(cuMemAlloc(&pointer_, bytes), "cuMemAlloc");
    try {
      checkCu(cuMemcpyHtoD(pointer_, source, bytes), "cuMemcpyHtoD");
    } catch (...) {
      reset();
      throw;
    }
  }

  DeviceAllocation(const DeviceAllocation &) = delete;
  DeviceAllocation &operator=(const DeviceAllocation &) = delete;
  DeviceAllocation(DeviceAllocation &&other) noexcept
      : pointer_(std::exchange(other.pointer_, 0)) {}
  DeviceAllocation &operator=(DeviceAllocation &&other) noexcept {
    if (this != &other) {
      reset();
      pointer_ = std::exchange(other.pointer_, 0);
    }
    return *this;
  }
  ~DeviceAllocation() { reset(); }

  CUdeviceptr get() const { return pointer_; }

private:
  void reset() noexcept {
    if (pointer_ != 0) {
      reportCleanupError(cuMemFree(pointer_), "cuMemFree");
      pointer_ = 0;
    }
  }

  CUdeviceptr pointer_ = 0;
};

static std::string readFile(std::string_view path) {
  std::ifstream file(std::string(path), std::ios::binary);
  if (!file)
    throw std::runtime_error("cannot open " + std::string(path));
  std::ostringstream contents;
  contents << file.rdbuf();
  return contents.str();
}

template <typename T>
static void printOutput(const ArrayParam<T> &array, CUdeviceptr devicePointer,
                        size_t count) {
  std::vector<T> output(array.values.size());
  checkCu(cuMemcpyDtoH(output.data(), devicePointer,
                       array.values.size() * sizeof(T)),
          "cuMemcpyDtoH");
  for (size_t i = 0; i < count; ++i) {
    if (i > 0)
      std::cout << ",";
    if constexpr (std::is_floating_point_v<T>)
      std::cout << std::setprecision(17) << output[i];
    else
      std::cout << output[i];
  }
  std::cout << "\n";
}
#endif

static int run(int argc, char **argv) {
  const char *ptxFile = nullptr;
  const char *kernelName = nullptr;
  std::vector<Param> params;
  Dims grid, block;
  int outputParam = 0;
  std::optional<int> outputCount;

  for (int i = 1; i < argc; ++i) {
    std::string_view argument = argv[i];
    auto needsValue = [&](std::string_view option) {
      if (++i >= argc)
        throw std::runtime_error(std::string(option) + " requires a value");
    };
    if (argument == "--param") {
      needsValue("--param");
      params.push_back(parseParam(argv[i]));
    } else if (argument == "--grid") {
      needsValue("--grid");
      grid = parseDims(argv[i]);
    } else if (argument == "--block") {
      needsValue("--block");
      block = parseDims(argv[i]);
    } else if (argument == "--output-param") {
      needsValue("--output-param");
      outputParam = parseIntArg(argv[i], "--output-param");
    } else if (argument == "--output-count") {
      needsValue("--output-count");
      outputCount = parseIntArg(argv[i], "--output-count");
    } else if (argument == "--kernel") {
      needsValue("--kernel");
      kernelName = argv[i];
    } else if (!argument.empty() && argument.front() == '-') {
      throw std::runtime_error("unknown option " + std::string(argument));
    } else {
      ptxFile = argv[i];
    }
  }

  if (!ptxFile) {
    std::cerr << "Usage: warpforth-runner kernel.ptx --kernel NAME "
                 "[--param i64[]:V,...] [--param f64[]:V,...] "
                 "[--param i64:V] [--param f64:V] [--grid X,Y,Z] "
                 "[--block X,Y,Z] [--output-param N] [--output-count N]\n";
    return 1;
  }
  if (!kernelName)
    throw std::runtime_error("--kernel NAME is required");
  if (params.empty())
    throw std::runtime_error("at least one --param is required");
  if (outputParam < 0 || static_cast<size_t>(outputParam) >= params.size()) {
    std::ostringstream message;
    message << "output-param " << outputParam << " out of range (have "
            << params.size() << " params)";
    throw std::runtime_error(message.str());
  }
  if (isScalar(params[outputParam])) {
    std::ostringstream message;
    message << "output-param " << outputParam
            << " is a scalar (cannot read back)";
    throw std::runtime_error(message.str());
  }

  size_t selectedArraySize = arraySize(params[outputParam]);
  if (outputCount && *outputCount < 0)
    throw std::runtime_error("--output-count must not be negative");
  if (outputCount && static_cast<size_t>(*outputCount) > selectedArraySize) {
    std::ostringstream message;
    message << "--output-count " << *outputCount
            << " exceeds selected output array size " << selectedArraySize;
    throw std::runtime_error(message.str());
  }

#ifdef WARPFORTH_RUNNER_HOST_VALIDATION
  std::cerr << "Error: CUDA execution unavailable in host-validation build\n";
  return 2;
#else
  size_t count =
      outputCount ? static_cast<size_t>(*outputCount) : selectedArraySize;
  std::string ptx = readFile(ptxFile);
  checkCu(cuInit(0), "cuInit");

  CUdevice device;
  checkCu(cuDeviceGet(&device, 0), "cuDeviceGet");
  CudaContext context = CudaContext::create(device);
  CudaModule module = CudaModule::load(ptx.c_str());

  CUfunction function;
  checkCu(cuModuleGetFunction(&function, module.get(), kernelName),
          "cuModuleGetFunction");

  std::vector<DeviceAllocation> allocations;
  allocations.reserve(params.size());
  std::vector<CUdeviceptr> devicePointers(params.size(), 0);
  for (size_t i = 0; i < params.size(); ++i) {
    if (auto *array = std::get_if<ArrayParam<int64_t>>(&params[i])) {
      allocations.emplace_back(array->values.data(),
                               array->values.size() * sizeof(int64_t));
      devicePointers[i] = allocations.back().get();
    } else if (auto *array = std::get_if<ArrayParam<double>>(&params[i])) {
      allocations.emplace_back(array->values.data(),
                               array->values.size() * sizeof(double));
      devicePointers[i] = allocations.back().get();
    }
  }

  std::vector<void *> kernelArguments(params.size());
  for (size_t i = 0; i < params.size(); ++i) {
    if (std::holds_alternative<ArrayParam<int64_t>>(params[i]) ||
        std::holds_alternative<ArrayParam<double>>(params[i]))
      kernelArguments[i] = &devicePointers[i];
    else if (auto *scalar = std::get_if<ScalarParam<int64_t>>(&params[i]))
      kernelArguments[i] = &scalar->value;
    else
      kernelArguments[i] = &std::get<ScalarParam<double>>(params[i]).value;
  }

  checkCu(cuLaunchKernel(function, grid.x, grid.y, grid.z, block.x, block.y,
                         block.z, 0, nullptr, kernelArguments.data(), nullptr),
          "cuLaunchKernel");
  checkCu(cuCtxSynchronize(), "cuCtxSynchronize");

  CUdeviceptr outputPointer = devicePointers[outputParam];
  if (auto *array = std::get_if<ArrayParam<int64_t>>(&params[outputParam]))
    printOutput(*array, outputPointer, count);
  else
    printOutput(std::get<ArrayParam<double>>(params[outputParam]),
                outputPointer, count);
  return 0;
#endif
}

int main(int argc, char **argv) {
  try {
    return run(argc, argv);
  } catch (const std::exception &error) {
    std::cerr << "Error: " << error.what() << "\n";
    return 1;
  }
}
