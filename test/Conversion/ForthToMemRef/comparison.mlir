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
    %1 = forth.literal %0 1 : !forth.stack -> !forth.stack
    %2 = forth.literal %1 2 : !forth.stack -> !forth.stack
    %3 = forth.eq %2 : !forth.stack -> !forth.stack
    %4 = forth.literal %3 3 : !forth.stack -> !forth.stack
    %5 = forth.literal %4 4 : !forth.stack -> !forth.stack
    %6 = forth.lt %5 : !forth.stack -> !forth.stack
    %7 = forth.literal %6 5 : !forth.stack -> !forth.stack
    %8 = forth.literal %7 6 : !forth.stack -> !forth.stack
    %9 = forth.gt %8 : !forth.stack -> !forth.stack
    %10 = forth.literal %9 0 : !forth.stack -> !forth.stack
    %11 = forth.zero_eq %10 : !forth.stack -> !forth.stack
    %12 = forth.literal %11 7 : !forth.stack -> !forth.stack
    %13 = forth.literal %12 8 : !forth.stack -> !forth.stack
    %14 = forth.ne %13 : !forth.stack -> !forth.stack
    %15 = forth.literal %14 9 : !forth.stack -> !forth.stack
    %16 = forth.literal %15 10 : !forth.stack -> !forth.stack
    %17 = forth.le %16 : !forth.stack -> !forth.stack
    %18 = forth.literal %17 11 : !forth.stack -> !forth.stack
    %19 = forth.literal %18 12 : !forth.stack -> !forth.stack
    %20 = forth.ge %19 : !forth.stack -> !forth.stack
    return
  }
}
