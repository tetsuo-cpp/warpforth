// RUN: %warpforth-opt --lower-forth-stack %s | %FileCheck %s

// Each GPU index word pushes its forth.intrinsic as an i64 cell, and
// GLOBAL-ID is BID-X * BDIM-X + TID-X. ForthToGPU lowers the intrinsics.
// CHECK-LABEL: func.func @gpu_indices(
// CHECK: %[[TID_X:.*]] = forth.intrinsic "tid-x" : index
// CHECK: %[[TID_X_CELL:.*]] = arith.index_cast %[[TID_X]] : index to i64
// CHECK: llvm.store %[[TID_X_CELL]],
// CHECK: %[[TID_Y:.*]] = forth.intrinsic "tid-y" : index
// CHECK: %[[TID_Y_CELL:.*]] = arith.index_cast %[[TID_Y]] : index to i64
// CHECK: llvm.store %[[TID_Y_CELL]],
// CHECK: %[[TID_Z:.*]] = forth.intrinsic "tid-z" : index
// CHECK: %[[TID_Z_CELL:.*]] = arith.index_cast %[[TID_Z]] : index to i64
// CHECK: llvm.store %[[TID_Z_CELL]],
// CHECK: %[[BID_X:.*]] = forth.intrinsic "bid-x" : index
// CHECK: %[[BID_X_CELL:.*]] = arith.index_cast %[[BID_X]] : index to i64
// CHECK: llvm.store %[[BID_X_CELL]],
// CHECK: %[[BID_Y:.*]] = forth.intrinsic "bid-y" : index
// CHECK: %[[BID_Y_CELL:.*]] = arith.index_cast %[[BID_Y]] : index to i64
// CHECK: llvm.store %[[BID_Y_CELL]],
// CHECK: %[[BID_Z:.*]] = forth.intrinsic "bid-z" : index
// CHECK: %[[BID_Z_CELL:.*]] = arith.index_cast %[[BID_Z]] : index to i64
// CHECK: llvm.store %[[BID_Z_CELL]],
// CHECK: %[[BDIM_X:.*]] = forth.intrinsic "bdim-x" : index
// CHECK: %[[BDIM_X_CELL:.*]] = arith.index_cast %[[BDIM_X]] : index to i64
// CHECK: llvm.store %[[BDIM_X_CELL]],
// CHECK: %[[BDIM_Y:.*]] = forth.intrinsic "bdim-y" : index
// CHECK: %[[BDIM_Y_CELL:.*]] = arith.index_cast %[[BDIM_Y]] : index to i64
// CHECK: llvm.store %[[BDIM_Y_CELL]],
// CHECK: %[[BDIM_Z:.*]] = forth.intrinsic "bdim-z" : index
// CHECK: %[[BDIM_Z_CELL:.*]] = arith.index_cast %[[BDIM_Z]] : index to i64
// CHECK: llvm.store %[[BDIM_Z_CELL]],
// CHECK: %[[GDIM_X:.*]] = forth.intrinsic "gdim-x" : index
// CHECK: %[[GDIM_X_CELL:.*]] = arith.index_cast %[[GDIM_X]] : index to i64
// CHECK: llvm.store %[[GDIM_X_CELL]],
// CHECK: %[[GDIM_Y:.*]] = forth.intrinsic "gdim-y" : index
// CHECK: %[[GDIM_Y_CELL:.*]] = arith.index_cast %[[GDIM_Y]] : index to i64
// CHECK: llvm.store %[[GDIM_Y_CELL]],
// CHECK: %[[GDIM_Z:.*]] = forth.intrinsic "gdim-z" : index
// CHECK: %[[GDIM_Z_CELL:.*]] = arith.index_cast %[[GDIM_Z]] : index to i64
// CHECK: llvm.store %[[GDIM_Z_CELL]],
// CHECK: %[[BLOCK:.*]] = forth.intrinsic "bid-x" : index
// CHECK: %[[DIM:.*]] = forth.intrinsic "bdim-x" : index
// CHECK: %[[THREAD:.*]] = forth.intrinsic "tid-x" : index
// CHECK: %[[OFFSET:.*]] = arith.muli %[[BLOCK]], %[[DIM]] : index
// CHECK: %[[GLOBAL:.*]] = arith.addi %[[OFFSET]], %[[THREAD]] : index
// CHECK: %[[GLOBAL_CELL:.*]] = arith.index_cast %[[GLOBAL]] : index to i64
// CHECK: llvm.store %[[GLOBAL_CELL]],
func.func @gpu_indices(%out: memref<1xi64>) {
  %s0 = forth.stack !forth.stack
  %s1 = forth.thread_id_x %s0 : !forth.stack -> !forth.stack
  %s2 = forth.param_ref %s1 %out : !forth.stack, memref<1xi64> -> !forth.stack
  %s3 = forth.storei %s2 : !forth.stack -> !forth.stack
  %s4 = forth.thread_id_y %s3 : !forth.stack -> !forth.stack
  %s5 = forth.param_ref %s4 %out : !forth.stack, memref<1xi64> -> !forth.stack
  %s6 = forth.storei %s5 : !forth.stack -> !forth.stack
  %s7 = forth.thread_id_z %s6 : !forth.stack -> !forth.stack
  %s8 = forth.param_ref %s7 %out : !forth.stack, memref<1xi64> -> !forth.stack
  %s9 = forth.storei %s8 : !forth.stack -> !forth.stack
  %s10 = forth.block_id_x %s9 : !forth.stack -> !forth.stack
  %s11 = forth.param_ref %s10 %out : !forth.stack, memref<1xi64> -> !forth.stack
  %s12 = forth.storei %s11 : !forth.stack -> !forth.stack
  %s13 = forth.block_id_y %s12 : !forth.stack -> !forth.stack
  %s14 = forth.param_ref %s13 %out : !forth.stack, memref<1xi64> -> !forth.stack
  %s15 = forth.storei %s14 : !forth.stack -> !forth.stack
  %s16 = forth.block_id_z %s15 : !forth.stack -> !forth.stack
  %s17 = forth.param_ref %s16 %out : !forth.stack, memref<1xi64> -> !forth.stack
  %s18 = forth.storei %s17 : !forth.stack -> !forth.stack
  %s19 = forth.block_dim_x %s18 : !forth.stack -> !forth.stack
  %s20 = forth.param_ref %s19 %out : !forth.stack, memref<1xi64> -> !forth.stack
  %s21 = forth.storei %s20 : !forth.stack -> !forth.stack
  %s22 = forth.block_dim_y %s21 : !forth.stack -> !forth.stack
  %s23 = forth.param_ref %s22 %out : !forth.stack, memref<1xi64> -> !forth.stack
  %s24 = forth.storei %s23 : !forth.stack -> !forth.stack
  %s25 = forth.block_dim_z %s24 : !forth.stack -> !forth.stack
  %s26 = forth.param_ref %s25 %out : !forth.stack, memref<1xi64> -> !forth.stack
  %s27 = forth.storei %s26 : !forth.stack -> !forth.stack
  %s28 = forth.grid_dim_x %s27 : !forth.stack -> !forth.stack
  %s29 = forth.param_ref %s28 %out : !forth.stack, memref<1xi64> -> !forth.stack
  %s30 = forth.storei %s29 : !forth.stack -> !forth.stack
  %s31 = forth.grid_dim_y %s30 : !forth.stack -> !forth.stack
  %s32 = forth.param_ref %s31 %out : !forth.stack, memref<1xi64> -> !forth.stack
  %s33 = forth.storei %s32 : !forth.stack -> !forth.stack
  %s34 = forth.grid_dim_z %s33 : !forth.stack -> !forth.stack
  %s35 = forth.param_ref %s34 %out : !forth.stack, memref<1xi64> -> !forth.stack
  %s36 = forth.storei %s35 : !forth.stack -> !forth.stack
  %s37 = forth.global_id %s36 : !forth.stack -> !forth.stack
  %s38 = forth.param_ref %s37 %out : !forth.stack, memref<1xi64> -> !forth.stack
  %s39 = forth.storei %s38 : !forth.stack -> !forth.stack
  return
}
