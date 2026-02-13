// RUN: %warpforth-opt --convert-forth-to-memref %s | %FileCheck %s

// CHECK-LABEL: func.func private @main

// load (@): pop address, inttoptr, llvm.load, store back
// CHECK: memref.load %{{.*}}[%{{.*}}] : memref<256xi64>
// CHECK: llvm.inttoptr %{{.*}} : i64 to !llvm.ptr
// CHECK: llvm.load %{{.*}} : !llvm.ptr -> i64
// CHECK: memref.store %{{.*}}, %{{.*}}[%{{.*}}] : memref<256xi64>

// store (!): pop address, pop value, inttoptr, llvm.store
// CHECK: memref.load
// CHECK: arith.subi
// CHECK: memref.load
// CHECK: llvm.inttoptr %{{.*}} : i64 to !llvm.ptr
// CHECK: llvm.store %{{.*}}, %{{.*}} : i64, !llvm.ptr

module {
  func.func private @main() {
    %0 = forth.stack !forth.stack
    %1 = forth.literal %0 1 : !forth.stack -> !forth.stack
    %2 = forth.load %1 : !forth.stack -> !forth.stack
    %3 = forth.literal %2 42 : !forth.stack -> !forth.stack
    %4 = forth.literal %3 100 : !forth.stack -> !forth.stack
    %5 = forth.store %4 : !forth.stack -> !forth.stack
    return
  }
}
