// RUN: %warpforth-opt --convert-forth-to-gpu %s | %FileCheck %s

// CHECK: gpu.module @warpforth_module
// CHECK: gpu.func @main() kernel
// CHECK: memref.alloca() : memref<256xi64>
// CHECK: gpu.return

module {
  func.func private @main() {
    %alloca = memref.alloca() : memref<256xi64>
    %c0 = arith.constant 0 : index
    return
  }
}
