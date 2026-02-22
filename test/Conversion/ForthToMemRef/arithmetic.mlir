// RUN: %warpforth-opt --convert-forth-to-memref %s | %FileCheck %s

// CHECK-LABEL: func.func private @main

// add: pop two, trunci to i32, arith.addi i32, extsi to i64, store
// CHECK: memref.load
// CHECK: arith.subi
// CHECK: memref.load
// CHECK: arith.trunci %{{.*}} : i64 to i32
// CHECK: arith.trunci %{{.*}} : i64 to i32
// CHECK: arith.addi %{{.*}}, %{{.*}} : i32
// CHECK: arith.extsi %{{.*}} : i32 to i64
// CHECK: memref.store

// sub: trunci, subi i32, extsi
// CHECK: memref.load
// CHECK: memref.load
// CHECK: arith.trunci %{{.*}} : i64 to i32
// CHECK: arith.trunci %{{.*}} : i64 to i32
// CHECK: arith.subi %{{.*}}, %{{.*}} : i32
// CHECK: arith.extsi %{{.*}} : i32 to i64
// CHECK: memref.store

// mul: trunci, muli i32, extsi
// CHECK: memref.load
// CHECK: memref.load
// CHECK: arith.trunci %{{.*}} : i64 to i32
// CHECK: arith.trunci %{{.*}} : i64 to i32
// CHECK: arith.muli %{{.*}}, %{{.*}} : i32
// CHECK: arith.extsi %{{.*}} : i32 to i64
// CHECK: memref.store

// div: trunci, divsi i32, extsi
// CHECK: memref.load
// CHECK: memref.load
// CHECK: arith.trunci %{{.*}} : i64 to i32
// CHECK: arith.trunci %{{.*}} : i64 to i32
// CHECK: arith.divsi %{{.*}}, %{{.*}} : i32
// CHECK: arith.extsi %{{.*}} : i32 to i64
// CHECK: memref.store

// mod: trunci, remsi i32, extsi
// CHECK: memref.load
// CHECK: memref.load
// CHECK: arith.trunci %{{.*}} : i64 to i32
// CHECK: arith.trunci %{{.*}} : i64 to i32
// CHECK: arith.remsi %{{.*}}, %{{.*}} : i32
// CHECK: arith.extsi %{{.*}} : i32 to i64
// CHECK: memref.store

module {
  func.func private @main() {
    %0 = forth.stack !forth.stack
    %1 = forth.constant %0(10 : i32) : !forth.stack -> !forth.stack
    %2 = forth.constant %1(20 : i32) : !forth.stack -> !forth.stack
    %3 = forth.addi %2 : !forth.stack -> !forth.stack
    %4 = forth.constant %3(3 : i32) : !forth.stack -> !forth.stack
    %5 = forth.subi %4 : !forth.stack -> !forth.stack
    %6 = forth.constant %5(4 : i32) : !forth.stack -> !forth.stack
    %7 = forth.muli %6 : !forth.stack -> !forth.stack
    %8 = forth.constant %7(2 : i32) : !forth.stack -> !forth.stack
    %9 = forth.divi %8 : !forth.stack -> !forth.stack
    %10 = forth.constant %9(5 : i32) : !forth.stack -> !forth.stack
    %11 = forth.mod %10 : !forth.stack -> !forth.stack
    return
  }
}
