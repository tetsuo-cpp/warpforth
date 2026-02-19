// RUN: %warpforth-opt --convert-forth-to-gpu %s | %FileCheck %s

// CHECK: gpu.module @warpforth_module
// CHECK: gpu.func @main() kernel

// CHECK: gpu.thread_id x
// CHECK: gpu.thread_id y
// CHECK: gpu.block_id z
// CHECK: gpu.block_dim x
// CHECK: gpu.grid_dim y

module {
  func.func private @main() attributes {forth.kernel} {
    %alloca = memref.alloca() : memref<256xi64>
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %0 = arith.addi %c0, %c1 : index
    %1 = forth.intrinsic "tid-x" : index
    %2 = arith.index_cast %1 : index to i64
    memref.store %2, %alloca[%0] : memref<256xi64>
    %3 = arith.addi %0, %c1 : index
    %4 = forth.intrinsic "tid-y" : index
    %5 = arith.index_cast %4 : index to i64
    memref.store %5, %alloca[%3] : memref<256xi64>
    %6 = arith.addi %3, %c1 : index
    %7 = forth.intrinsic "bid-z" : index
    %8 = arith.index_cast %7 : index to i64
    memref.store %8, %alloca[%6] : memref<256xi64>
    %9 = arith.addi %6, %c1 : index
    %10 = forth.intrinsic "bdim-x" : index
    %11 = arith.index_cast %10 : index to i64
    memref.store %11, %alloca[%9] : memref<256xi64>
    %12 = arith.addi %9, %c1 : index
    %13 = forth.intrinsic "gdim-y" : index
    %14 = arith.index_cast %13 : index to i64
    memref.store %14, %alloca[%12] : memref<256xi64>
    return
  }
}
