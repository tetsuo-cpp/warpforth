get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)

add_llvm_executable(warpforth-translate
  warpforth-translate.cpp
  )

llvm_update_compile_flags(warpforth-translate)

target_link_libraries(warpforth-translate
  PRIVATE
  ${dialect_libs}
  MLIRForth
  MLIRForthTranslation
  MLIRFuncDialect
  MLIRIR
  MLIRSupport
  )

mlir_check_all_link_libraries(warpforth-translate)
