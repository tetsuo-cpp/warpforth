// RUN: %warpforth-opt --convert-forth-to-memref %s | %FileCheck %s

// CHECK-LABEL: func.func private @main(%{{.*}}: memref<256xi32> {forth.param_name = "data"})
// CHECK: memref.alloca() : memref<256xi64>
// CHECK: memref.extract_aligned_pointer_as_index %{{.*}} : memref<256xi32> -> index
// CHECK: arith.index_cast %{{.*}} : index to i64
// CHECK: memref.store %{{.*}}, %{{.*}}[%{{.*}}] : memref<256xi64>

module {
  func.func private @main(%arg0: memref<256xi32> {forth.param_name = "data"}) {
    %0 = forth.stack !forth.stack
    %1 = forth.param_ref %0 "data" : !forth.stack -> !forth.stack
    return
  }
}
