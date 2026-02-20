// RUN: %warpforth-opt --convert-forth-to-memref %s | %FileCheck %s

// CHECK-LABEL: func.func private @main

// addf: pop two, bitcast to f64, arith.addf, bitcast back, store
// CHECK: memref.load
// CHECK: arith.subi
// CHECK: memref.load
// CHECK: arith.bitcast %{{.*}} : i64 to f64
// CHECK: arith.bitcast %{{.*}} : i64 to f64
// CHECK: arith.addf %{{.*}}, %{{.*}} : f64
// CHECK: arith.bitcast %{{.*}} : f64 to i64
// CHECK: memref.store

// subf: bitcast, subf, bitcast
// CHECK: arith.bitcast %{{.*}} : i64 to f64
// CHECK: arith.bitcast %{{.*}} : i64 to f64
// CHECK: arith.subf %{{.*}}, %{{.*}} : f64
// CHECK: arith.bitcast %{{.*}} : f64 to i64

// mulf: bitcast, mulf, bitcast
// CHECK: arith.bitcast %{{.*}} : i64 to f64
// CHECK: arith.bitcast %{{.*}} : i64 to f64
// CHECK: arith.mulf %{{.*}}, %{{.*}} : f64
// CHECK: arith.bitcast %{{.*}} : f64 to i64

// divf: bitcast, divf, bitcast
// CHECK: arith.bitcast %{{.*}} : i64 to f64
// CHECK: arith.bitcast %{{.*}} : i64 to f64
// CHECK: arith.divf %{{.*}}, %{{.*}} : f64
// CHECK: arith.bitcast %{{.*}} : f64 to i64

module {
  func.func private @main() {
    %0 = forth.stack !forth.stack
    %1 = forth.constant %0(1.000000e+00 : f64) : !forth.stack -> !forth.stack
    %2 = forth.constant %1(2.000000e+00 : f64) : !forth.stack -> !forth.stack
    %3 = forth.addf %2 : !forth.stack -> !forth.stack
    %4 = forth.subf %3 : !forth.stack -> !forth.stack
    %5 = forth.mulf %4 : !forth.stack -> !forth.stack
    %6 = forth.divf %5 : !forth.stack -> !forth.stack
    return
  }
}
