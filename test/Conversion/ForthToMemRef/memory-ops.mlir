// RUN: %warpforth-opt --convert-forth-to-memref %s | %FileCheck %s

// CHECK-LABEL: func.func private @main

// load (@): pop address, inttoptr, llvm.load i32, extsi i32->i64, store back
// CHECK: memref.load %{{.*}}[%{{.*}}] : memref<256xi64>
// CHECK: llvm.inttoptr %{{.*}} : i64 to !llvm.ptr
// CHECK: llvm.load %{{.*}} : !llvm.ptr -> i32
// CHECK: arith.extsi %{{.*}} : i32 to i64
// CHECK: memref.store %{{.*}}, %{{.*}}[%{{.*}}] : memref<256xi64>

// store (!): pop address, pop value, trunci i64->i32, inttoptr, llvm.store i32
// CHECK: memref.load
// CHECK: arith.subi
// CHECK: memref.load
// CHECK: arith.trunci %{{.*}} : i64 to i32
// CHECK: llvm.inttoptr %{{.*}} : i64 to !llvm.ptr
// CHECK: llvm.store %{{.*}}, %{{.*}} : i32, !llvm.ptr

// shared load (S@): pop address, inttoptr shared addrspace, llvm.load i32, extsi
// CHECK: llvm.inttoptr %{{.*}} : i64 to !llvm.ptr<3>
// CHECK: llvm.load %{{.*}} : !llvm.ptr<3> -> i32
// CHECK: arith.extsi %{{.*}} : i32 to i64

// shared store (S!): pop address + value, trunci, inttoptr shared addrspace, llvm.store i32
// CHECK: arith.trunci %{{.*}} : i64 to i32
// CHECK: llvm.inttoptr %{{.*}} : i64 to !llvm.ptr<3>
// CHECK: llvm.store %{{.*}}, %{{.*}} : i32, !llvm.ptr<3>

module {
  func.func private @main() {
    %0 = forth.stack !forth.stack
    %1 = forth.constant %0(1 : i32) : !forth.stack -> !forth.stack
    %2 = forth.loadi %1 : !forth.stack -> !forth.stack
    %3 = forth.constant %2(42 : i32) : !forth.stack -> !forth.stack
    %4 = forth.constant %3(100 : i32) : !forth.stack -> !forth.stack
    %5 = forth.storei %4 : !forth.stack -> !forth.stack
    %6 = forth.constant %5(2 : i32) : !forth.stack -> !forth.stack
    %7 = forth.shared_loadi %6 : !forth.stack -> !forth.stack
    %8 = forth.constant %7(9 : i32) : !forth.stack -> !forth.stack
    %9 = forth.constant %8(3 : i32) : !forth.stack -> !forth.stack
    %10 = forth.shared_storei %9 : !forth.stack -> !forth.stack
    return
  }
}
