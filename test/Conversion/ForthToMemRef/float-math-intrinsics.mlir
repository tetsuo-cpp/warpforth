// RUN: %warpforth-opt --convert-forth-to-memref %s | %FileCheck %s

// CHECK-LABEL: func.func private @main

// expf: load, bitcast i64->f64, math.exp, bitcast f64->i64, store (SP unchanged)
// CHECK: memref.load
// CHECK: arith.bitcast %{{.*}} : i64 to f64
// CHECK: math.exp %{{.*}} : f64
// CHECK: arith.bitcast %{{.*}} : f64 to i64
// CHECK: memref.store

// sqrtf: load, bitcast, math.sqrt, bitcast, store
// CHECK: memref.load
// CHECK: arith.bitcast %{{.*}} : i64 to f64
// CHECK: math.sqrt %{{.*}} : f64
// CHECK: arith.bitcast %{{.*}} : f64 to i64
// CHECK: memref.store

// logf: load, bitcast, math.log, bitcast, store
// CHECK: memref.load
// CHECK: arith.bitcast %{{.*}} : i64 to f64
// CHECK: math.log %{{.*}} : f64
// CHECK: arith.bitcast %{{.*}} : f64 to i64
// CHECK: memref.store

// absf: load, bitcast, math.absf, bitcast, store
// CHECK: memref.load
// CHECK: arith.bitcast %{{.*}} : i64 to f64
// CHECK: math.absf %{{.*}} : f64
// CHECK: arith.bitcast %{{.*}} : f64 to i64
// CHECK: memref.store

// negf: load, bitcast, arith.negf, bitcast, store
// CHECK: memref.load
// CHECK: arith.bitcast %{{.*}} : i64 to f64
// CHECK: arith.negf %{{.*}} : f64
// CHECK: arith.bitcast %{{.*}} : f64 to i64
// CHECK: memref.store

// maxf: binary — pop two, bitcast, arith.maximumf, bitcast, store
// CHECK: memref.load
// CHECK: arith.subi
// CHECK: memref.load
// CHECK: arith.bitcast %{{.*}} : i64 to f64
// CHECK: arith.bitcast %{{.*}} : i64 to f64
// CHECK: arith.maximumf %{{.*}}, %{{.*}} : f64
// CHECK: arith.bitcast %{{.*}} : f64 to i64
// CHECK: memref.store

// minf: binary — pop two, bitcast, arith.minimumf, bitcast, store
// CHECK: arith.bitcast %{{.*}} : i64 to f64
// CHECK: arith.bitcast %{{.*}} : i64 to f64
// CHECK: arith.minimumf %{{.*}}, %{{.*}} : f64
// CHECK: arith.bitcast %{{.*}} : f64 to i64

module {
  func.func private @main() {
    %0 = forth.stack !forth.stack
    %1 = forth.constant %0(1.000000e+00 : f64) : !forth.stack -> !forth.stack
    %2 = forth.expf %1 : !forth.stack -> !forth.stack
    %3 = forth.sqrtf %2 : !forth.stack -> !forth.stack
    %4 = forth.logf %3 : !forth.stack -> !forth.stack
    %5 = forth.absf %4 : !forth.stack -> !forth.stack
    %6 = forth.negf %5 : !forth.stack -> !forth.stack
    %7 = forth.constant %6(2.000000e+00 : f64) : !forth.stack -> !forth.stack
    %8 = forth.maxf %7 : !forth.stack -> !forth.stack
    %9 = forth.minf %8 : !forth.stack -> !forth.stack
    return
  }
}
