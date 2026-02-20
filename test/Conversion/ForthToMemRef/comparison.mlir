// RUN: %warpforth-opt --convert-forth-to-memref %s | %FileCheck %s

// CHECK-LABEL: func.func private @main

// eq: load two values, arith.cmpi eq, extsi to i64, store
// CHECK: memref.load
// CHECK: arith.subi
// CHECK: memref.load
// CHECK: arith.cmpi eq, %{{.*}}, %{{.*}} : i64
// CHECK: arith.extsi %{{.*}} : i1 to i64
// CHECK: memref.store

// lt: load two values, arith.cmpi slt, extsi to i64, store
// CHECK: memref.load
// CHECK: memref.load
// CHECK: arith.cmpi slt, %{{.*}}, %{{.*}} : i64
// CHECK: arith.extsi %{{.*}} : i1 to i64
// CHECK: memref.store

// gt: load two values, arith.cmpi sgt, extsi to i64, store
// CHECK: memref.load
// CHECK: memref.load
// CHECK: arith.cmpi sgt, %{{.*}}, %{{.*}} : i64
// CHECK: arith.extsi %{{.*}} : i1 to i64
// CHECK: memref.store

// zero_eq: load one value, compare with 0, extsi, store at same SP
// CHECK: memref.load
// CHECK: arith.constant 0 : i64
// CHECK: arith.cmpi eq, %{{.*}}, %{{.*}} : i64
// CHECK: arith.extsi %{{.*}} : i1 to i64
// CHECK: memref.store

// ne: load two values, arith.cmpi ne, extsi to i64, store
// CHECK: memref.load
// CHECK: memref.load
// CHECK: arith.cmpi ne, %{{.*}}, %{{.*}} : i64
// CHECK: arith.extsi %{{.*}} : i1 to i64
// CHECK: memref.store

// le: load two values, arith.cmpi sle, extsi to i64, store
// CHECK: memref.load
// CHECK: memref.load
// CHECK: arith.cmpi sle, %{{.*}}, %{{.*}} : i64
// CHECK: arith.extsi %{{.*}} : i1 to i64
// CHECK: memref.store

// ge: load two values, arith.cmpi sge, extsi to i64, store
// CHECK: memref.load
// CHECK: memref.load
// CHECK: arith.cmpi sge, %{{.*}}, %{{.*}} : i64
// CHECK: arith.extsi %{{.*}} : i1 to i64
// CHECK: memref.store

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
