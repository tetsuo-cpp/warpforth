// RUN: %warpforth-opt --convert-forth-to-memref %s | %FileCheck %s

// CHECK-LABEL: func.func private @array_param(
// CHECK-SAME: %[[ARRAY:.*]]: memref<256xi64> {forth.param_name = "data"})
// CHECK-NOT: memref.alloca() : memref<256xi64>
// CHECK: %[[POINTER:.*]] = memref.extract_aligned_pointer_as_index %[[ARRAY]] : memref<256xi64> -> index
// CHECK: arith.index_cast %[[POINTER]] : index to i64

// CHECK-LABEL: func.func private @i64_param(
// CHECK-SAME: %{{.*}}: i64 {forth.param_name = "count"})
// CHECK-NOT: memref.alloca() : memref<256xi64>
// CHECK: return

// CHECK-LABEL: func.func private @f64_param(
// CHECK-SAME: %[[FLOAT:.*]]: f64 {forth.param_name = "scale"})
// CHECK-NOT: memref.alloca() : memref<256xi64>
// CHECK: arith.bitcast %[[FLOAT]] : f64 to i64

module {
  func.func private @array_param(%arg0: memref<256xi64> {forth.param_name = "data"}) {
    %0 = forth.stack !forth.stack
    %1 = forth.param_ref %0 %arg0 : !forth.stack, memref<256xi64> -> !forth.stack
    return
  }
  func.func private @i64_param(%arg0: i64 {forth.param_name = "count"}) {
    %0 = forth.stack !forth.stack
    %1 = forth.param_ref %0 %arg0 : !forth.stack, i64 -> !forth.stack
    return
  }
  func.func private @f64_param(%arg0: f64 {forth.param_name = "scale"}) {
    %0 = forth.stack !forth.stack
    %1 = forth.param_ref %0 %arg0 : !forth.stack, f64 -> !forth.stack
    return
  }
}
