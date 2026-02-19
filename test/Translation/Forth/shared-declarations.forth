\ RUN: %warpforth-translate --forth-to-mlir %s | %FileCheck %s

\ Verify shared memory declarations produce tagged alloca and pointer push sequence
\ CHECK: func.func private @main(%arg0: memref<256xi64> {forth.param_name = "DATA"})
\ CHECK: memref.alloca() {forth.shared_name = "SCRATCH"} : memref<256xi64>
\ CHECK: memref.extract_aligned_pointer_as_index
\ CHECK: arith.index_cast
\ CHECK: forth.push_value
\! kernel main
\! param DATA i64[256]
\! shared SCRATCH i64[256]
SCRATCH
