// RUN: %warpforth-opt --convert-forth-to-memref %s | %FileCheck %s

// CHECK-LABEL: func.func private @main

// Verify flag load and condition:
// CHECK: %[[FLAG:.*]] = memref.load %{{.*}}[%[[SP:.*]]]
// CHECK: %[[ZERO:.*]] = arith.constant 0 : i64
// CHECK: %[[COND:.*]] = arith.cmpi ne, %[[FLAG]], %[[ZERO]] : i64

// Verify scf.if with index result:
// CHECK: scf.if %[[COND]] -> (index) {

// Then branch: drop (subi) + literal push + yield
// CHECK:   arith.subi
// CHECK:   arith.constant 42 : i64
// CHECK:   arith.addi
// CHECK:   memref.store
// CHECK:   scf.yield %{{.*}} : index

// Else branch: drop (subi) + literal push + yield
// CHECK: } else {
// CHECK:   arith.subi
// CHECK:   arith.constant 99 : i64
// CHECK:   arith.addi
// CHECK:   memref.store
// CHECK:   scf.yield %{{.*}} : index
// CHECK: }

module {
  func.func private @main() {
    %0 = forth.stack !forth.stack
    %1 = forth.literal %0 1 : !forth.stack -> !forth.stack
    %2 = forth.if %1 : !forth.stack -> !forth.stack {
    ^bb0(%arg0: !forth.stack):
      %3 = forth.drop %arg0 : !forth.stack -> !forth.stack
      %4 = forth.literal %3 42 : !forth.stack -> !forth.stack
      forth.yield %4 : !forth.stack
    } else {
    ^bb0(%arg0: !forth.stack):
      %3 = forth.drop %arg0 : !forth.stack -> !forth.stack
      %4 = forth.literal %3 99 : !forth.stack -> !forth.stack
      forth.yield %4 : !forth.stack
    }
    return
  }
}
