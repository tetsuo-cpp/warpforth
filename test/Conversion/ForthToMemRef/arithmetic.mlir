// RUN: %warpforth-opt --convert-forth-to-memref %s | %FileCheck %s

// CHECK-LABEL: func.func private @main

// add: pop two, arith.addi, store result
// CHECK: memref.load
// CHECK: arith.subi
// CHECK: memref.load
// CHECK: arith.addi %{{.*}}, %{{.*}} : i64
// CHECK: memref.store

// sub: pop two, arith.subi
// CHECK: memref.load
// CHECK: memref.load
// CHECK: arith.subi %{{.*}}, %{{.*}} : i64
// CHECK: memref.store

// mul: pop two, arith.muli
// CHECK: memref.load
// CHECK: memref.load
// CHECK: arith.muli %{{.*}}, %{{.*}} : i64
// CHECK: memref.store

// div: pop two, arith.divsi
// CHECK: memref.load
// CHECK: memref.load
// CHECK: arith.divsi %{{.*}}, %{{.*}} : i64
// CHECK: memref.store

// mod: pop two, arith.remsi
// CHECK: memref.load
// CHECK: memref.load
// CHECK: arith.remsi %{{.*}}, %{{.*}} : i64
// CHECK: memref.store

module {
  func.func private @main() {
    %0 = forth.stack !forth.stack
    %1 = forth.literal %0 10 : !forth.stack -> !forth.stack
    %2 = forth.literal %1 20 : !forth.stack -> !forth.stack
    %3 = forth.add %2 : !forth.stack -> !forth.stack
    %4 = forth.literal %3 3 : !forth.stack -> !forth.stack
    %5 = forth.sub %4 : !forth.stack -> !forth.stack
    %6 = forth.literal %5 4 : !forth.stack -> !forth.stack
    %7 = forth.mul %6 : !forth.stack -> !forth.stack
    %8 = forth.literal %7 2 : !forth.stack -> !forth.stack
    %9 = forth.div %8 : !forth.stack -> !forth.stack
    %10 = forth.literal %9 5 : !forth.stack -> !forth.stack
    %11 = forth.mod %10 : !forth.stack -> !forth.stack
    return
  }
}
