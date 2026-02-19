// RUN: %warpforth-opt --convert-forth-to-gpu %s | %FileCheck %s

// CHECK: gpu.module @warpforth_module
// CHECK: gpu.func @main(%arg0: memref<256xi64> {forth.param_name = "DATA"})
// CHECK-SAME: workgroup(%{{.*}}: memref<256xi64, #gpu.address_space<workgroup>>)
// CHECK-SAME: kernel
// CHECK-NOT: memref.alloca() {forth.shared_name
// CHECK: memref.extract_aligned_pointer_as_index %{{.*}} : memref<256xi64, #gpu.address_space<workgroup>>
// CHECK: gpu.return

module {
  func.func private @main(%arg0: memref<256xi64> {forth.param_name = "DATA"}) attributes {forth.kernel} {
    %alloca = memref.alloca() {forth.shared_name = "SCRATCH"} : memref<256xi64>
    %ptr = memref.extract_aligned_pointer_as_index %alloca : memref<256xi64> -> index
    %c0 = arith.constant 0 : index
    return
  }
}
