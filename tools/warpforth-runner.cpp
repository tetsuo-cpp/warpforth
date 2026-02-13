#include <cstdlib>
#include <cuda.h>
#include <cuda_runtime.h>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#define BUFFER_SIZE 256

std::vector<int64_t> parseValues(const std::string &str) {
  std::vector<int64_t> values;
  std::stringstream ss(str);
  std::string item;

  while (std::getline(ss, item, ',')) {
    values.push_back(std::stoll(item));
  }

  return values;
}

void printUsage(const char *programName) {
  std::cerr
      << "Usage: " << programName
      << " <ptx_file> [--param values|N] ... [--output-param idx] "
         "[--output-count N]\n"
      << "\n"
      << "  --param values   Add a parameter with comma-separated initial "
         "values\n"
      << "  --param N        Add a parameter initialized with N zeros\n"
      << "  --output-param   Which parameter index to print (default: 0)\n"
      << "  --output-count   How many elements to print (default: 10)\n";
}

int main(int argc, char **argv) {
  if (argc < 2) {
    printUsage(argv[0]);
    return 1;
  }

  std::string ptxFile = argv[1];
  int outputCount = 10;
  int outputParam = 0;

  // Each param is a vector of initial values (padded to BUFFER_SIZE with 0s)
  std::vector<std::vector<int64_t>> params;

  // Parse command line arguments
  for (int i = 2; i < argc; i++) {
    std::string arg = argv[i];

    if (arg == "--param" && i + 1 < argc) {
      std::string val = argv[++i];
      // Check if it's a plain number (size) or comma-separated values
      bool isPlainNumber = true;
      for (char c : val) {
        if (c == ',') {
          isPlainNumber = false;
          break;
        }
      }
      if (isPlainNumber && val.find_first_not_of("0123456789") == std::string::npos) {
        int size = std::stoi(val);
        params.push_back(std::vector<int64_t>(size, 0));
      } else {
        params.push_back(parseValues(val));
      }
    } else if (arg == "--output-param" && i + 1 < argc) {
      outputParam = std::stoi(argv[++i]);
    } else if (arg == "--output-count" && i + 1 < argc) {
      outputCount = std::stoi(argv[++i]);
      if (outputCount < 0 || outputCount > BUFFER_SIZE) {
        std::cerr << "Error: output-count must be between 0 and " << BUFFER_SIZE
                  << "\n";
        return 1;
      }
    } else {
      std::cerr << "Error: Unknown argument " << arg << "\n";
      printUsage(argv[0]);
      return 1;
    }
  }

  // Default: one zero-initialized parameter if none specified
  if (params.empty()) {
    params.push_back(std::vector<int64_t>(BUFFER_SIZE, 0));
  }

  if (outputParam < 0 || outputParam >= (int)params.size()) {
    std::cerr << "Error: output-param " << outputParam << " out of range (have "
              << params.size() << " params)\n";
    return 1;
  }

  std::ifstream file(ptxFile);
  if (!file) {
    std::cerr << "Error: Cannot open PTX file " << argv[1] << "\n";
    return 1;
  }

  std::string ptxCode((std::istreambuf_iterator<char>(file)),
                      std::istreambuf_iterator<char>());

  CUcontext context;
  CUdevice device;
  cuInit(0);
  cuDeviceGet(&device, 0);
  cuCtxCreate(&context, 0, device);

  CUmodule module;
  cuModuleLoadDataEx(&module, ptxCode.c_str(), 0, nullptr, nullptr);

  CUfunction kernel;
  cuModuleGetFunction(&kernel, module, "main");

  // Allocate and initialize device buffers for each param
  std::vector<int64_t *> deviceBuffers(params.size());
  std::vector<std::vector<int64_t>> hostBuffers(params.size());

  for (size_t p = 0; p < params.size(); ++p) {
    hostBuffers[p].resize(BUFFER_SIZE, 0);
    for (size_t j = 0; j < params[p].size() && j < BUFFER_SIZE; ++j) {
      hostBuffers[p][j] = params[p][j];
    }
    cudaMalloc(&deviceBuffers[p], BUFFER_SIZE * sizeof(int64_t));
    cudaMemcpy(deviceBuffers[p], hostBuffers[p].data(),
               BUFFER_SIZE * sizeof(int64_t), cudaMemcpyHostToDevice);
  }

  // Build kernel args array (each element is a pointer to the device pointer)
  std::vector<void *> args(params.size());
  for (size_t p = 0; p < params.size(); ++p) {
    args[p] = &deviceBuffers[p];
  }

  cuLaunchKernel(kernel, 1, 1, 1, 1, 1, 1, 0, nullptr, args.data(), nullptr);

  // Copy back the output param
  cudaMemcpy(hostBuffers[outputParam].data(), deviceBuffers[outputParam],
             BUFFER_SIZE * sizeof(int64_t), cudaMemcpyDeviceToHost);

  for (int i = 0; i < outputCount; i++) {
    if (i > 0)
      std::cout << ",";
    std::cout << hostBuffers[outputParam][i];
  }
  std::cout << "\n";

  for (auto *buf : deviceBuffers) {
    cudaFree(buf);
  }
  cuModuleUnload(module);
  cuCtxDestroy(context);

  return 0;
}
