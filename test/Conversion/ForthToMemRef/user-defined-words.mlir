// RUN: %warpforth-opt --convert-forth-to-memref %s | %FileCheck %s

// User-defined word signature: (!forth.stack) -> !forth.stack becomes
// (memref<256xi64>, index) -> (memref<256xi64>, index)
// CHECK-LABEL: func.func private @double(%{{.*}}: memref<256xi64>, %{{.*}}: index) -> (memref<256xi64>, index)
// CHECK: memref.load
// CHECK: arith.addi
// CHECK: memref.store
// CHECK: return %{{.*}}, %{{.*}} : memref<256xi64>, index

// CHECK-LABEL: func.func private @main
// CHECK: call @double(%{{.*}}, %{{.*}}) : (memref<256xi64>, index) -> (memref<256xi64>, index)

module {
  func.func private @double(%arg0: !forth.stack) -> !forth.stack {
    %0 = forth.dup %arg0 : !forth.stack -> !forth.stack
    %1 = forth.addi %0 : !forth.stack -> !forth.stack
    return %1 : !forth.stack
  }
  func.func private @main() {
    %0 = forth.stack !forth.stack
    %1 = forth.constant %0(5 : i64) : !forth.stack -> !forth.stack
    %2 = call @double(%1) : (!forth.stack) -> !forth.stack
    return
  }
}
