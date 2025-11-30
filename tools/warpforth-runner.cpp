#include <cuda.h>
#include <cuda_runtime.h>
#include <fstream>
#include <iostream>
#include <string>

#define BUFFER_SIZE 256

int main(int argc, char **argv) {
  if (argc != 2) {
    std::cerr << "Usage: " << argv[0] << " <ptx_file>\n";
    return 1;
  }

  std::ifstream file(argv[1]);
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

  int64_t *deviceBuffer;
  cudaMalloc(&deviceBuffer, BUFFER_SIZE * sizeof(int64_t));
  cudaMemset(deviceBuffer, 0, BUFFER_SIZE * sizeof(int64_t));

  void *args[] = {&deviceBuffer};
  cuLaunchKernel(kernel, 1, 1, 1, 1, 1, 1, 0, nullptr, args, nullptr);

  int64_t hostBuffer[BUFFER_SIZE];
  cudaMemcpy(hostBuffer, deviceBuffer, BUFFER_SIZE * sizeof(int64_t),
             cudaMemcpyDeviceToHost);

  for (int i = 0; i < 10; i++) {
    std::cout << "buffer[" << i << "] = " << hostBuffer[i] << "\n";
  }

  cudaFree(deviceBuffer);
  cuModuleUnload(module);
  cuCtxDestroy(context);

  return 0;
}
