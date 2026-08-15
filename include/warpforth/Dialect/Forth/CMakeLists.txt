add_mlir_dialect(ForthOps forth)

set(LLVM_TARGET_DEFINITIONS ForthOps.td)
mlir_tablegen(ForthDialect.md -gen-dialect-doc)
add_public_tablegen_target(ForthDialectDocGen)
