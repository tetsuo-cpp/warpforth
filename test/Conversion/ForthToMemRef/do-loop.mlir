// RUN: %warpforth-opt --convert-forth-to-memref %s | %FileCheck %s

// CHECK-LABEL: func.func private @main

// Verify start and limit are popped from stack:
// CHECK: memref.load
// CHECK: arith.subi
// CHECK: memref.load
// CHECK: arith.subi

// Verify scf.for with index bounds and iter arg:
// CHECK: arith.index_cast
// CHECK: arith.index_cast
// CHECK: scf.for %{{.*}} = %{{.*}} to %{{.*}} step %{{.*}} iter_args(%{{.*}} = %{{.*}}) -> (index) {

// Verify loop_index pushes induction variable:
// CHECK:   arith.index_cast
// CHECK:   arith.addi
// CHECK:   memref.store
// CHECK:   scf.yield %{{.*}} : index
// CHECK: }

module {
  func.func private @main() {
    %0 = forth.stack !forth.stack
    %1 = forth.literal %0 10 : !forth.stack -> !forth.stack
    %2 = forth.literal %1 0 : !forth.stack -> !forth.stack
    %3 = forth.do_loop %2 : !forth.stack -> !forth.stack {
    ^bb0(%arg0: !forth.stack):
      %4 = forth.loop_index %arg0 : !forth.stack -> !forth.stack
      forth.yield %4 : !forth.stack
    }
    return
  }
}
