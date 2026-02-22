// RUN: %warpforth-opt --convert-forth-to-memref %s | %FileCheck %s

// CHECK-LABEL: func.func private @main

// addf: pop two, trunci to i32, bitcast i32->f32, arith.addf f32, bitcast f32->i32, extsi to i64, store
// CHECK: memref.load
// CHECK: arith.subi
// CHECK: memref.load
// CHECK: arith.trunci %{{.*}} : i64 to i32
// CHECK: arith.trunci %{{.*}} : i64 to i32
// CHECK: arith.bitcast %{{.*}} : i32 to f32
// CHECK: arith.bitcast %{{.*}} : i32 to f32
// CHECK: arith.addf %{{.*}}, %{{.*}} : f32
// CHECK: arith.bitcast %{{.*}} : f32 to i32
// CHECK: arith.extsi %{{.*}} : i32 to i64
// CHECK: memref.store

// subf: trunci, bitcast, subf f32, bitcast, extsi
// CHECK: arith.trunci %{{.*}} : i64 to i32
// CHECK: arith.trunci %{{.*}} : i64 to i32
// CHECK: arith.bitcast %{{.*}} : i32 to f32
// CHECK: arith.bitcast %{{.*}} : i32 to f32
// CHECK: arith.subf %{{.*}}, %{{.*}} : f32
// CHECK: arith.bitcast %{{.*}} : f32 to i32
// CHECK: arith.extsi %{{.*}} : i32 to i64

// mulf: trunci, bitcast, mulf f32, bitcast, extsi
// CHECK: arith.trunci %{{.*}} : i64 to i32
// CHECK: arith.trunci %{{.*}} : i64 to i32
// CHECK: arith.bitcast %{{.*}} : i32 to f32
// CHECK: arith.bitcast %{{.*}} : i32 to f32
// CHECK: arith.mulf %{{.*}}, %{{.*}} : f32
// CHECK: arith.bitcast %{{.*}} : f32 to i32
// CHECK: arith.extsi %{{.*}} : i32 to i64

// divf: trunci, bitcast, divf f32, bitcast, extsi
// CHECK: arith.trunci %{{.*}} : i64 to i32
// CHECK: arith.trunci %{{.*}} : i64 to i32
// CHECK: arith.bitcast %{{.*}} : i32 to f32
// CHECK: arith.bitcast %{{.*}} : i32 to f32
// CHECK: arith.divf %{{.*}}, %{{.*}} : f32
// CHECK: arith.bitcast %{{.*}} : f32 to i32
// CHECK: arith.extsi %{{.*}} : i32 to i64

module {
  func.func private @main() {
    %0 = forth.stack !forth.stack
    %1 = forth.constant %0(1.000000e+00 : f32) : !forth.stack -> !forth.stack
    %2 = forth.constant %1(2.000000e+00 : f32) : !forth.stack -> !forth.stack
    %3 = forth.addf %2 : !forth.stack -> !forth.stack
    %4 = forth.subf %3 : !forth.stack -> !forth.stack
    %5 = forth.mulf %4 : !forth.stack -> !forth.stack
    %6 = forth.divf %5 : !forth.stack -> !forth.stack
    return
  }
}
