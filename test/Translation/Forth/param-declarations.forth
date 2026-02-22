\ RUN: %warpforth-translate --forth-to-mlir %s | %FileCheck %s

\ Verify multi-param declarations with correct types and ordering
\ CHECK: func.func private @main(%arg0: memref<256xi32> {forth.param_name = "DATA"}, %arg1: memref<128xi32> {forth.param_name = "WEIGHTS"})
\ CHECK: forth.param_ref %{{.*}} "DATA"
\ CHECK: forth.param_ref %{{.*}} "WEIGHTS"
\! kernel main
\! param DATA i32[256]
\! param WEIGHTS i32[128]
DATA WEIGHTS
