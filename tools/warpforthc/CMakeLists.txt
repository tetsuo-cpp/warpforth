get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)
get_property(extension_libs GLOBAL PROPERTY MLIR_EXTENSION_LIBS)
get_property(translation_libs GLOBAL PROPERTY MLIR_TRANSLATION_LIBS)

add_llvm_executable(warpforthc
  warpforthc.cpp
)

llvm_update_compile_flags(warpforthc)

target_link_libraries(warpforthc
  PRIVATE
  ${dialect_libs}
  ${conversion_libs}
  ${extension_libs}
  ${translation_libs}
  MLIRForth
  MLIRForthTranslation
  MLIRToPTXTranslation
  MLIRConversionPasses
  MLIRForthToMemRefConversion
  MLIRNVVMTarget
  MLIRIR
  MLIRParser
  MLIRPass
  MLIRSupport
)

mlir_check_all_link_libraries(warpforthc)
