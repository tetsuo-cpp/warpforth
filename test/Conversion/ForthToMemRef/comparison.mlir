// RUN: %warpforth-opt --tag-forth-stack-depth --convert-forth-to-memref %s | %FileCheck %s

// CHECK-LABEL: func.func private @main
// CHECK-NOT: memref<256xi64>

// CHECK: %[[E1:.*]] = arith.constant 1 : i64
// CHECK: %[[E2:.*]] = arith.constant 2 : i64
// CHECK: %[[EQ:.*]] = arith.cmpi eq, %[[E1]], %[[E2]] : i64
// CHECK: arith.extsi %[[EQ]] : i1 to i64
// CHECK: %[[L3:.*]] = arith.constant 3 : i64
// CHECK: %[[L4:.*]] = arith.constant 4 : i64
// CHECK: %[[LT:.*]] = arith.cmpi slt, %[[L3]], %[[L4]] : i64
// CHECK: arith.extsi %[[LT]] : i1 to i64
// CHECK: %[[G5:.*]] = arith.constant 5 : i64
// CHECK: %[[G6:.*]] = arith.constant 6 : i64
// CHECK: %[[GT:.*]] = arith.cmpi sgt, %[[G5]], %[[G6]] : i64
// CHECK: arith.extsi %[[GT]] : i1 to i64
// CHECK: %[[Z:.*]] = arith.constant 0 : i64
// CHECK: %[[ZZ:.*]] = arith.constant 0 : i64
// CHECK: %[[ZE:.*]] = arith.cmpi eq, %[[Z]], %[[ZZ]] : i64
// CHECK: arith.extsi %[[ZE]] : i1 to i64
// CHECK: %[[N7:.*]] = arith.constant 7 : i64
// CHECK: %[[N8:.*]] = arith.constant 8 : i64
// CHECK: %[[NE:.*]] = arith.cmpi ne, %[[N7]], %[[N8]] : i64
// CHECK: arith.extsi %[[NE]] : i1 to i64
// CHECK: %[[LE9:.*]] = arith.constant 9 : i64
// CHECK: %[[LE10:.*]] = arith.constant 10 : i64
// CHECK: %[[LE:.*]] = arith.cmpi sle, %[[LE9]], %[[LE10]] : i64
// CHECK: arith.extsi %[[LE]] : i1 to i64
// CHECK: %[[GE11:.*]] = arith.constant 11 : i64
// CHECK: %[[GE12:.*]] = arith.constant 12 : i64
// CHECK: %[[GE:.*]] = arith.cmpi sge, %[[GE11]], %[[GE12]] : i64
// CHECK: arith.extsi %[[GE]] : i1 to i64
// CHECK-NOT: memref<256xi64>

module {
  func.func private @main() {
    %0 = forth.stack !forth.stack
    %1 = forth.constant %0(1 : i64) : !forth.stack -> !forth.stack
    %2 = forth.constant %1(2 : i64) : !forth.stack -> !forth.stack
    %3 = forth.eqi %2 : !forth.stack -> !forth.stack
    %4 = forth.constant %3(3 : i64) : !forth.stack -> !forth.stack
    %5 = forth.constant %4(4 : i64) : !forth.stack -> !forth.stack
    %6 = forth.lti %5 : !forth.stack -> !forth.stack
    %7 = forth.constant %6(5 : i64) : !forth.stack -> !forth.stack
    %8 = forth.constant %7(6 : i64) : !forth.stack -> !forth.stack
    %9 = forth.gti %8 : !forth.stack -> !forth.stack
    %10 = forth.constant %9(0 : i64) : !forth.stack -> !forth.stack
    %11 = forth.zero_eq %10 : !forth.stack -> !forth.stack
    %12 = forth.constant %11(7 : i64) : !forth.stack -> !forth.stack
    %13 = forth.constant %12(8 : i64) : !forth.stack -> !forth.stack
    %14 = forth.nei %13 : !forth.stack -> !forth.stack
    %15 = forth.constant %14(9 : i64) : !forth.stack -> !forth.stack
    %16 = forth.constant %15(10 : i64) : !forth.stack -> !forth.stack
    %17 = forth.lei %16 : !forth.stack -> !forth.stack
    %18 = forth.constant %17(11 : i64) : !forth.stack -> !forth.stack
    %19 = forth.constant %18(12 : i64) : !forth.stack -> !forth.stack
    %20 = forth.gei %19 : !forth.stack -> !forth.stack
    return
  }
}
