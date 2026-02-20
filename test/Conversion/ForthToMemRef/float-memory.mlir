// RUN: %warpforth-opt --convert-forth-to-memref %s | %FileCheck %s

// CHECK-LABEL: func.func private @main

// loadf: load addr, inttoptr, llvm.load f64, bitcast f64->i64, store
// CHECK: memref.load
// CHECK: llvm.inttoptr
// CHECK: llvm.load %{{.*}} : !llvm.ptr -> f64
// CHECK: arith.bitcast %{{.*}} : f64 to i64
// CHECK: memref.store

// storef: load addr, load value, inttoptr, bitcast i64->f64, llvm.store
// CHECK: memref.load
// CHECK: memref.load
// CHECK: llvm.inttoptr
// CHECK: arith.bitcast %{{.*}} : i64 to f64
// CHECK: llvm.store %{{.*}}, %{{.*}} : f64

module {
  func.func private @main() {
    %0 = forth.stack !forth.stack
    %1 = forth.constant %0(1000 : i64) : !forth.stack -> !forth.stack
    %2 = forth.loadf %1 : !forth.stack -> !forth.stack
    %3 = forth.constant %2(3.140000e+00 : f64) : !forth.stack -> !forth.stack
    %4 = forth.constant %3(2000 : i64) : !forth.stack -> !forth.stack
    %5 = forth.storef %4 : !forth.stack -> !forth.stack
    return
  }
}
