// RUN: %warpforth-opt --tag-forth-stack-depth --convert-forth-to-memref %s | %FileCheck %s

// CHECK-LABEL: func.func private @main
// CHECK-NOT: memref<256xi64>

// CHECK: %[[A3:.*]] = arith.constant 3 : i64
// CHECK: %[[A5:.*]] = arith.constant 5 : i64
// CHECK: arith.andi %[[A3]], %[[A5]] : i64
// CHECK: %[[O7:.*]] = arith.constant 7 : i64
// CHECK: %[[O8:.*]] = arith.constant 8 : i64
// CHECK: arith.ori %[[O7]], %[[O8]] : i64
// CHECK: %[[X15:.*]] = arith.constant 15 : i64
// CHECK: %[[X3:.*]] = arith.constant 3 : i64
// CHECK: arith.xori %[[X15]], %[[X3]] : i64
// CHECK: %[[N42:.*]] = arith.constant 42 : i64
// CHECK: %[[ONES:.*]] = arith.constant -1 : i64
// CHECK: arith.xori %[[N42]], %[[ONES]] : i64
// CHECK: %[[L1:.*]] = arith.constant 1 : i64
// CHECK: %[[L4:.*]] = arith.constant 4 : i64
// CHECK: arith.shli %[[L1]], %[[L4]] : i64
// CHECK: %[[R256:.*]] = arith.constant 256 : i64
// CHECK: %[[R2:.*]] = arith.constant 2 : i64
// CHECK: arith.shrui %[[R256]], %[[R2]] : i64
// CHECK-NOT: memref<256xi64>

module {
  func.func private @main() {
    %0 = forth.stack !forth.stack
    %1 = forth.constant %0(3 : i64) : !forth.stack -> !forth.stack
    %2 = forth.constant %1(5 : i64) : !forth.stack -> !forth.stack
    %3 = forth.and %2 : !forth.stack -> !forth.stack
    %4 = forth.constant %3(7 : i64) : !forth.stack -> !forth.stack
    %5 = forth.constant %4(8 : i64) : !forth.stack -> !forth.stack
    %6 = forth.or %5 : !forth.stack -> !forth.stack
    %7 = forth.constant %6(15 : i64) : !forth.stack -> !forth.stack
    %8 = forth.constant %7(3 : i64) : !forth.stack -> !forth.stack
    %9 = forth.xor %8 : !forth.stack -> !forth.stack
    %10 = forth.constant %9(42 : i64) : !forth.stack -> !forth.stack
    %11 = forth.not %10 : !forth.stack -> !forth.stack
    %12 = forth.constant %11(1 : i64) : !forth.stack -> !forth.stack
    %13 = forth.constant %12(4 : i64) : !forth.stack -> !forth.stack
    %14 = forth.lshift %13 : !forth.stack -> !forth.stack
    %15 = forth.constant %14(256 : i64) : !forth.stack -> !forth.stack
    %16 = forth.constant %15(2 : i64) : !forth.stack -> !forth.stack
    %17 = forth.rshift %16 : !forth.stack -> !forth.stack
    return
  }
}
