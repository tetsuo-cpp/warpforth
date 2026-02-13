// RUN: %warpforth-opt --convert-forth-to-gpu %s | %FileCheck %s

// CHECK: gpu.module @warpforth_module
// Main gets kernel attribute
// CHECK: gpu.func @main() kernel
// CHECK: gpu.return

// Private helper moved into gpu.module without kernel attr
// CHECK: func.func private @helper
// CHECK-NOT: kernel
// CHECK: return

module {
  func.func private @helper(%arg0: memref<256xi64>, %arg1: index) -> (memref<256xi64>, index) {
    return %arg0, %arg1 : memref<256xi64>, index
  }
  func.func private @main() {
    %alloca = memref.alloca() : memref<256xi64>
    %c0 = arith.constant 0 : index
    return
  }
}
