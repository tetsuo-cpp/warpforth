// RUN: %warpforth-opt --convert-forth-to-memref %s | %FileCheck %s

// CHECK-LABEL: func.func private @main

// expf: load, trunci to i32, bitcast i32->f32, math.exp f32, bitcast f32->i32, extsi to i64, store (SP unchanged)
// CHECK: memref.load
// CHECK: arith.trunci %{{.*}} : i64 to i32
// CHECK: arith.bitcast %{{.*}} : i32 to f32
// CHECK: math.exp %{{.*}} : f32
// CHECK: arith.bitcast %{{.*}} : f32 to i32
// CHECK: arith.extsi %{{.*}} : i32 to i64
// CHECK: memref.store

// sqrtf: trunci, bitcast, math.sqrt f32, bitcast, extsi
// CHECK: memref.load
// CHECK: arith.trunci %{{.*}} : i64 to i32
// CHECK: arith.bitcast %{{.*}} : i32 to f32
// CHECK: math.sqrt %{{.*}} : f32
// CHECK: arith.bitcast %{{.*}} : f32 to i32
// CHECK: arith.extsi %{{.*}} : i32 to i64
// CHECK: memref.store

// logf: trunci, bitcast, math.log f32, bitcast, extsi
// CHECK: memref.load
// CHECK: arith.trunci %{{.*}} : i64 to i32
// CHECK: arith.bitcast %{{.*}} : i32 to f32
// CHECK: math.log %{{.*}} : f32
// CHECK: arith.bitcast %{{.*}} : f32 to i32
// CHECK: arith.extsi %{{.*}} : i32 to i64
// CHECK: memref.store

// absf: trunci, bitcast, math.absf f32, bitcast, extsi
// CHECK: memref.load
// CHECK: arith.trunci %{{.*}} : i64 to i32
// CHECK: arith.bitcast %{{.*}} : i32 to f32
// CHECK: math.absf %{{.*}} : f32
// CHECK: arith.bitcast %{{.*}} : f32 to i32
// CHECK: arith.extsi %{{.*}} : i32 to i64
// CHECK: memref.store

// negf: trunci, bitcast, arith.negf f32, bitcast, extsi
// CHECK: memref.load
// CHECK: arith.trunci %{{.*}} : i64 to i32
// CHECK: arith.bitcast %{{.*}} : i32 to f32
// CHECK: arith.negf %{{.*}} : f32
// CHECK: arith.bitcast %{{.*}} : f32 to i32
// CHECK: arith.extsi %{{.*}} : i32 to i64
// CHECK: memref.store

// maxf: binary — pop two, trunci, bitcast, arith.maximumf f32, bitcast, extsi, store
// CHECK: memref.load
// CHECK: arith.subi
// CHECK: memref.load
// CHECK: arith.trunci %{{.*}} : i64 to i32
// CHECK: arith.trunci %{{.*}} : i64 to i32
// CHECK: arith.bitcast %{{.*}} : i32 to f32
// CHECK: arith.bitcast %{{.*}} : i32 to f32
// CHECK: arith.maximumf %{{.*}}, %{{.*}} : f32
// CHECK: arith.bitcast %{{.*}} : f32 to i32
// CHECK: arith.extsi %{{.*}} : i32 to i64
// CHECK: memref.store

// minf: binary — trunci, bitcast, arith.minimumf f32, bitcast, extsi
// CHECK: arith.trunci %{{.*}} : i64 to i32
// CHECK: arith.trunci %{{.*}} : i64 to i32
// CHECK: arith.bitcast %{{.*}} : i32 to f32
// CHECK: arith.bitcast %{{.*}} : i32 to f32
// CHECK: arith.minimumf %{{.*}}, %{{.*}} : f32
// CHECK: arith.bitcast %{{.*}} : f32 to i32
// CHECK: arith.extsi %{{.*}} : i32 to i64

module {
  func.func private @main() {
    %0 = forth.stack !forth.stack
    %1 = forth.constant %0(1.000000e+00 : f32) : !forth.stack -> !forth.stack
    %2 = forth.expf %1 : !forth.stack -> !forth.stack
    %3 = forth.sqrtf %2 : !forth.stack -> !forth.stack
    %4 = forth.logf %3 : !forth.stack -> !forth.stack
    %5 = forth.absf %4 : !forth.stack -> !forth.stack
    %6 = forth.negf %5 : !forth.stack -> !forth.stack
    %7 = forth.constant %6(2.000000e+00 : f32) : !forth.stack -> !forth.stack
    %8 = forth.maxf %7 : !forth.stack -> !forth.stack
    %9 = forth.minf %8 : !forth.stack -> !forth.stack
    return
  }
}
