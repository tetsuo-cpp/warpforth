find_package(CUDAToolkit QUIET)

add_executable(warpforth-runner-validate EXCLUDE_FROM_ALL warpforth-runner.cpp)
target_compile_definitions(warpforth-runner-validate
  PRIVATE WARPFORTH_RUNNER_HOST_VALIDATION)
target_compile_features(warpforth-runner-validate PRIVATE cxx_std_17)
set_target_properties(warpforth-runner-validate PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY ${LLVM_RUNTIME_OUTPUT_INTDIR})

if(CUDAToolkit_FOUND)
  add_executable(warpforth-runner warpforth-runner.cpp)
  target_link_libraries(warpforth-runner PRIVATE CUDA::cuda_driver)
  target_compile_features(warpforth-runner PRIVATE cxx_std_17)
else()
  message(STATUS "CUDAToolkit not found — skipping warpforth-runner")
endif()
