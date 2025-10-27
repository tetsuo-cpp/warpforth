get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)

add_llvm_executable(warpforth-opt
  warpforth-opt.cpp
  )

llvm_update_compile_flags(warpforth-opt)

target_link_libraries(warpforth-opt
  PRIVATE
  ${dialect_libs}
  ${conversion_libs}
  MLIRForth
  MLIRForthToGPU
  MLIROptLib
  MLIRPass
  MLIRTransforms
  )

mlir_check_all_link_libraries(warpforth-opt)
