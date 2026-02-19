\ RUN: %warpforth-translate --forth-to-mlir %s | %FileCheck %s

\ Verify multi-param declarations with correct types and ordering
\ CHECK: func.func private @main(%arg0: memref<256xi64> {forth.param_name = "DATA"}, %arg1: memref<128xi64> {forth.param_name = "WEIGHTS"})
\ CHECK: forth.param_ref %{{.*}} "DATA"
\ CHECK: forth.param_ref %{{.*}} "WEIGHTS"
\! kernel main
\! param DATA i64[256]
\! param WEIGHTS i64[128]
DATA WEIGHTS
