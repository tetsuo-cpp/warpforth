// RUN: %not %warpforth-translate --mlir-to-ptx %s 2>&1 | %FileCheck %s

// CHECK: no-binary-error.mlir:[[@LINE+2]]:1: error: no gpu.binary operations found in module
// CHECK-NOT: hint
module {
}
