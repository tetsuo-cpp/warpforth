set(LLVM_TARGET_DEFINITIONS ForthOps.td)
mlir_tablegen(ForthOps.h.inc -gen-op-decls)
mlir_tablegen(ForthOps.cpp.inc -gen-op-defs)
mlir_tablegen(ForthOpsDialect.h.inc -gen-dialect-decls)
mlir_tablegen(ForthOpsDialect.cpp.inc -gen-dialect-defs)
add_public_tablegen_target(MLIRForthOpsIncGen)

add_mlir_doc(ForthOps ForthOps Forth/ -gen-op-doc)
