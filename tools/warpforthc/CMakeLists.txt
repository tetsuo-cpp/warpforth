add_llvm_executable(warpforthc
  warpforthc.cpp
)

llvm_update_compile_flags(warpforthc)

target_link_libraries(warpforthc
  PRIVATE
  MLIRForthTranslation
  MLIRToPTXTranslation
  MLIRConversionPasses
  MLIRArithToLLVM
  MLIRBuiltinToLLVMIRTranslation
  MLIRControlFlowToLLVM
  MLIRFuncToLLVM
  MLIRGPUToLLVMIRTranslation
  MLIRLLVMToLLVMIRTranslation
  MLIRMemRefToLLVM
  MLIRNVVMToLLVMIRTranslation
  MLIRNVVMTarget
  MLIRIR
  MLIRPass
  MLIRSupport
)

mlir_check_all_link_libraries(warpforthc)
