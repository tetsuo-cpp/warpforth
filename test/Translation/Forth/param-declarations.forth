\ RUN: %warpforth-translate --forth-to-mlir %s | %FileCheck %s

\ Verify multi-param declarations with correct types and ordering
\ CHECK: func.func private @main(%arg0: memref<256xi64> {forth.param_name = "data"}, %arg1: memref<128xi64> {forth.param_name = "weights"})
\ CHECK: forth.param_ref %{{.*}} "data"
\ CHECK: forth.param_ref %{{.*}} "weights"
param data 256
param weights 128
data weights
