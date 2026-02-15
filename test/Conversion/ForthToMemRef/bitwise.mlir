// RUN: %warpforth-opt --convert-forth-to-memref %s | %FileCheck %s

// CHECK-LABEL: func.func private @main

// and: pop two, arith.andi, store result
// CHECK: memref.load
// CHECK: arith.subi
// CHECK: memref.load
// CHECK: arith.andi %{{.*}}, %{{.*}} : i64
// CHECK: memref.store

// or: pop two, arith.ori, store result
// CHECK: memref.load
// CHECK: memref.load
// CHECK: arith.ori %{{.*}}, %{{.*}} : i64
// CHECK: memref.store

// xor: pop two, arith.xori, store result
// CHECK: memref.load
// CHECK: memref.load
// CHECK: arith.xori %{{.*}}, %{{.*}} : i64
// CHECK: memref.store

// not: load one value, xori with -1, store at same SP
// CHECK: memref.load
// CHECK: arith.constant -1 : i64
// CHECK: arith.xori %{{.*}}, %{{.*}} : i64
// CHECK: memref.store

// lshift: pop two, arith.shli, store result
// CHECK: memref.load
// CHECK: memref.load
// CHECK: arith.shli %{{.*}}, %{{.*}} : i64
// CHECK: memref.store

// rshift: pop two, arith.shrui, store result
// CHECK: memref.load
// CHECK: memref.load
// CHECK: arith.shrui %{{.*}}, %{{.*}} : i64
// CHECK: memref.store

module {
  func.func private @main() {
    %0 = forth.stack !forth.stack
    %1 = forth.literal %0 3 : !forth.stack -> !forth.stack
    %2 = forth.literal %1 5 : !forth.stack -> !forth.stack
    %3 = forth.and %2 : !forth.stack -> !forth.stack
    %4 = forth.literal %3 7 : !forth.stack -> !forth.stack
    %5 = forth.literal %4 8 : !forth.stack -> !forth.stack
    %6 = forth.or %5 : !forth.stack -> !forth.stack
    %7 = forth.literal %6 15 : !forth.stack -> !forth.stack
    %8 = forth.literal %7 3 : !forth.stack -> !forth.stack
    %9 = forth.xor %8 : !forth.stack -> !forth.stack
    %10 = forth.literal %9 42 : !forth.stack -> !forth.stack
    %11 = forth.not %10 : !forth.stack -> !forth.stack
    %12 = forth.literal %11 1 : !forth.stack -> !forth.stack
    %13 = forth.literal %12 4 : !forth.stack -> !forth.stack
    %14 = forth.lshift %13 : !forth.stack -> !forth.stack
    %15 = forth.literal %14 256 : !forth.stack -> !forth.stack
    %16 = forth.literal %15 2 : !forth.stack -> !forth.stack
    %17 = forth.rshift %16 : !forth.stack -> !forth.stack
    return
  }
}
