add_mlir_dialect(ForthOps forth)

set(LLVM_TARGET_DEFINITIONS ForthOps.td)
mlir_tablegen(ForthOpsEnums.h.inc -gen-enum-decls)
mlir_tablegen(ForthOpsEnums.cpp.inc -gen-enum-defs)
add_public_tablegen_target(MLIRForthOpsEnumsIncGen)

set(LLVM_TARGET_DEFINITIONS ForthOps.td)
mlir_tablegen(ForthDialect.md -gen-dialect-doc)
add_public_tablegen_target(ForthDialectDocGen)
