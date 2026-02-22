// RUN: %warpforth-opt --convert-forth-to-memref %s | %FileCheck %s

// CHECK-LABEL: func.func private @main
// CHECK: memref.alloca() : memref<256xi64>
// CHECK: arith.constant 0 : index
// CHECK: arith.constant 42 : i32
// CHECK: arith.extsi %{{.*}} : i32 to i64
// CHECK: arith.constant 1 : index
// CHECK: arith.addi %{{.*}}, %{{.*}} : index
// CHECK: memref.store %{{.*}}, %{{.*}}[%{{.*}}] : memref<256xi64>

module {
  func.func private @main() {
    %0 = forth.stack !forth.stack
    %1 = forth.constant %0(42 : i32) : !forth.stack -> !forth.stack
    return
  }
}
