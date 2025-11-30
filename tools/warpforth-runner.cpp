#include <cstdlib>
#include <cuda.h>
#include <cuda_runtime.h>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#define BUFFER_SIZE 256

std::vector<int64_t> parseInputs(const std::string &inputStr) {
  std::vector<int64_t> inputs;
  std::stringstream ss(inputStr);
  std::string item;

  while (std::getline(ss, item, ',')) {
    inputs.push_back(std::stoll(item));
  }

  return inputs;
}

void printUsage(const char *programName) {
  std::cerr << "Usage: " << programName
            << " <ptx_file> [--inputs val1,val2,...] [--output-count N]\n";
}

int main(int argc, char **argv) {
  if (argc < 2) {
    printUsage(argv[0]);
    return 1;
  }

  std::string ptxFile = argv[1];
  std::vector<int64_t> inputs;
  int outputCount = 10;

  // Parse command line arguments
  for (int i = 2; i < argc; i++) {
    std::string arg = argv[i];

    if (arg == "--inputs" && i + 1 < argc) {
      inputs = parseInputs(argv[++i]);
      if (inputs.size() > BUFFER_SIZE) {
        std::cerr << "Error: Too many input values. Maximum allowed is "
                  << BUFFER_SIZE << "\n";
        return 1;
      }
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

  int64_t hostBuffer[BUFFER_SIZE];

  // Initialize buffer with input values
  for (int i = 0; i < BUFFER_SIZE; i++) {
    if (i < inputs.size()) {
      hostBuffer[i] = inputs[i];
    } else {
      hostBuffer[i] = 0;
    }
  }

  int64_t *deviceBuffer;
  cudaMalloc(&deviceBuffer, BUFFER_SIZE * sizeof(int64_t));
  cudaMemcpy(deviceBuffer, hostBuffer, BUFFER_SIZE * sizeof(int64_t),
             cudaMemcpyHostToDevice);

  void *args[] = {&deviceBuffer};
  cuLaunchKernel(kernel, 1, 1, 1, 1, 1, 1, 0, nullptr, args, nullptr);

  cudaMemcpy(hostBuffer, deviceBuffer, BUFFER_SIZE * sizeof(int64_t),
             cudaMemcpyDeviceToHost);

  for (int i = 0; i < outputCount; i++) {
    if (i > 0)
      std::cout << ",";
    std::cout << hostBuffer[i];
  }
  std::cout << "\n";

  cudaFree(deviceBuffer);
  cuModuleUnload(module);
  cuCtxDestroy(context);

  return 0;
}
