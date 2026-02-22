// RUN: %warpforth-opt --convert-forth-to-memref %s | %FileCheck %s

// CHECK-LABEL: func.func private @main

// loadf: load addr, inttoptr, llvm.load f32, bitcast f32->i32, extsi i32->i64, store
// CHECK: memref.load
// CHECK: llvm.inttoptr
// CHECK: llvm.load %{{.*}} : !llvm.ptr -> f32
// CHECK: arith.bitcast %{{.*}} : f32 to i32
// CHECK: arith.extsi %{{.*}} : i32 to i64
// CHECK: memref.store

// storef: load addr, load value, trunci i64->i32, inttoptr, bitcast i32->f32, llvm.store f32
// CHECK: memref.load
// CHECK: memref.load
// CHECK: arith.trunci %{{.*}} : i64 to i32
// CHECK: llvm.inttoptr
// CHECK: arith.bitcast %{{.*}} : i32 to f32
// CHECK: llvm.store %{{.*}}, %{{.*}} : f32

module {
  func.func private @main() {
    %0 = forth.stack !forth.stack
    %1 = forth.constant %0(1000 : i32) : !forth.stack -> !forth.stack
    %2 = forth.loadf %1 : !forth.stack -> !forth.stack
    %3 = forth.constant %2(3.140000e+00 : f32) : !forth.stack -> !forth.stack
    %4 = forth.constant %3(2000 : i32) : !forth.stack -> !forth.stack
    %5 = forth.storef %4 : !forth.stack -> !forth.stack
    return
  }
}
