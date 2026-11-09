// RUN: %warpforth-opt --tag-forth-stack-depth --convert-forth-to-memref %s | %FileCheck %s

// CHECK-LABEL: func.func private @main
// CHECK-NOT: memref<256xi64>

// CHECK: %[[C10:.*]] = arith.constant 10 : i64
// CHECK: %[[C20:.*]] = arith.constant 20 : i64
// CHECK: %[[ADD:.*]] = arith.addi %[[C10]], %[[C20]] : i64
// CHECK: %[[C3:.*]] = arith.constant 3 : i64
// CHECK: %[[SUB:.*]] = arith.subi %[[ADD]], %[[C3]] : i64
// CHECK: %[[C4:.*]] = arith.constant 4 : i64
// CHECK: %[[MUL:.*]] = arith.muli %[[SUB]], %[[C4]] : i64
// CHECK: %[[C2:.*]] = arith.constant 2 : i64
// CHECK: %[[DIV:.*]] = arith.divsi %[[MUL]], %[[C2]] : i64
// CHECK: %[[C5:.*]] = arith.constant 5 : i64
// CHECK: arith.remsi %[[DIV]], %[[C5]] : i64
// CHECK-NOT: memref<256xi64>

module {
  func.func private @main() {
    %0 = forth.stack !forth.stack
    %1 = forth.constant %0(10 : i64) : !forth.stack -> !forth.stack
    %2 = forth.constant %1(20 : i64) : !forth.stack -> !forth.stack
    %3 = forth.addi %2 : !forth.stack -> !forth.stack
    %4 = forth.constant %3(3 : i64) : !forth.stack -> !forth.stack
    %5 = forth.subi %4 : !forth.stack -> !forth.stack
    %6 = forth.constant %5(4 : i64) : !forth.stack -> !forth.stack
    %7 = forth.muli %6 : !forth.stack -> !forth.stack
    %8 = forth.constant %7(2 : i64) : !forth.stack -> !forth.stack
    %9 = forth.divi %8 : !forth.stack -> !forth.stack
    %10 = forth.constant %9(5 : i64) : !forth.stack -> !forth.stack
    %11 = forth.mod %10 : !forth.stack -> !forth.stack
    return
  }
}
