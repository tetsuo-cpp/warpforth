// RUN: %warpforth-opt --convert-forth-to-memref %s | %FileCheck %s

// CHECK-LABEL: func.func private @main

// eq: load two, trunci to i32, cmpi eq on i32, extsi i1->i64, store
// CHECK: memref.load
// CHECK: arith.subi
// CHECK: memref.load
// CHECK: arith.trunci %{{.*}} : i64 to i32
// CHECK: arith.trunci %{{.*}} : i64 to i32
// CHECK: arith.cmpi eq, %{{.*}}, %{{.*}} : i32
// CHECK: arith.extsi %{{.*}} : i1 to i64
// CHECK: memref.store

// lt: trunci, cmpi slt on i32, extsi
// CHECK: memref.load
// CHECK: memref.load
// CHECK: arith.trunci %{{.*}} : i64 to i32
// CHECK: arith.trunci %{{.*}} : i64 to i32
// CHECK: arith.cmpi slt, %{{.*}}, %{{.*}} : i32
// CHECK: arith.extsi %{{.*}} : i1 to i64
// CHECK: memref.store

// gt: trunci, cmpi sgt on i32, extsi
// CHECK: memref.load
// CHECK: memref.load
// CHECK: arith.trunci %{{.*}} : i64 to i32
// CHECK: arith.trunci %{{.*}} : i64 to i32
// CHECK: arith.cmpi sgt, %{{.*}}, %{{.*}} : i32
// CHECK: arith.extsi %{{.*}} : i1 to i64
// CHECK: memref.store

// zero_eq: trunci to i32, compare with 0:i32, extsi i1->i64, store
// CHECK: memref.load
// CHECK: arith.trunci %{{.*}} : i64 to i32
// CHECK: arith.constant 0 : i32
// CHECK: arith.cmpi eq, %{{.*}}, %{{.*}} : i32
// CHECK: arith.extsi %{{.*}} : i1 to i64
// CHECK: memref.store

// ne: trunci, cmpi ne on i32, extsi
// CHECK: memref.load
// CHECK: memref.load
// CHECK: arith.trunci %{{.*}} : i64 to i32
// CHECK: arith.trunci %{{.*}} : i64 to i32
// CHECK: arith.cmpi ne, %{{.*}}, %{{.*}} : i32
// CHECK: arith.extsi %{{.*}} : i1 to i64
// CHECK: memref.store

// le: trunci, cmpi sle on i32, extsi
// CHECK: memref.load
// CHECK: memref.load
// CHECK: arith.trunci %{{.*}} : i64 to i32
// CHECK: arith.trunci %{{.*}} : i64 to i32
// CHECK: arith.cmpi sle, %{{.*}}, %{{.*}} : i32
// CHECK: arith.extsi %{{.*}} : i1 to i64
// CHECK: memref.store

// ge: trunci, cmpi sge on i32, extsi
// CHECK: memref.load
// CHECK: memref.load
// CHECK: arith.trunci %{{.*}} : i64 to i32
// CHECK: arith.trunci %{{.*}} : i64 to i32
// CHECK: arith.cmpi sge, %{{.*}}, %{{.*}} : i32
// CHECK: arith.extsi %{{.*}} : i1 to i64
// CHECK: memref.store

module {
  func.func private @main() {
    %0 = forth.stack !forth.stack
    %1 = forth.constant %0(1 : i32) : !forth.stack -> !forth.stack
    %2 = forth.constant %1(2 : i32) : !forth.stack -> !forth.stack
    %3 = forth.eqi %2 : !forth.stack -> !forth.stack
    %4 = forth.constant %3(3 : i32) : !forth.stack -> !forth.stack
    %5 = forth.constant %4(4 : i32) : !forth.stack -> !forth.stack
    %6 = forth.lti %5 : !forth.stack -> !forth.stack
    %7 = forth.constant %6(5 : i32) : !forth.stack -> !forth.stack
    %8 = forth.constant %7(6 : i32) : !forth.stack -> !forth.stack
    %9 = forth.gti %8 : !forth.stack -> !forth.stack
    %10 = forth.constant %9(0 : i32) : !forth.stack -> !forth.stack
    %11 = forth.zero_eq %10 : !forth.stack -> !forth.stack
    %12 = forth.constant %11(7 : i32) : !forth.stack -> !forth.stack
    %13 = forth.constant %12(8 : i32) : !forth.stack -> !forth.stack
    %14 = forth.nei %13 : !forth.stack -> !forth.stack
    %15 = forth.constant %14(9 : i32) : !forth.stack -> !forth.stack
    %16 = forth.constant %15(10 : i32) : !forth.stack -> !forth.stack
    %17 = forth.lei %16 : !forth.stack -> !forth.stack
    %18 = forth.constant %17(11 : i32) : !forth.stack -> !forth.stack
    %19 = forth.constant %18(12 : i32) : !forth.stack -> !forth.stack
    %20 = forth.gei %19 : !forth.stack -> !forth.stack
    return
  }
}
