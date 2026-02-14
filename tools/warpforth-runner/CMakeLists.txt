find_package(CUDAToolkit QUIET)

if(CUDAToolkit_FOUND)
  add_executable(warpforth-runner warpforth-runner.cpp)
  target_link_libraries(warpforth-runner PRIVATE CUDA::cuda_driver)
  target_compile_features(warpforth-runner PRIVATE cxx_std_17)
else()
  message(STATUS "CUDAToolkit not found — skipping warpforth-runner")
endif()
