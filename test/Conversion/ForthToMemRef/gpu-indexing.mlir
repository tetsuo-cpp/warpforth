// RUN: %warpforth-opt --convert-forth-to-memref %s | %FileCheck %s

// CHECK-LABEL: func.func private @main
// CHECK: %[[TIDX:.*]] = gpu.thread_id x
// CHECK: %[[TIDX64:.*]] = arith.index_cast %[[TIDX]] : index to i64
// CHECK: memref.store %[[TIDX64]],
// CHECK: gpu.thread_id y
// CHECK: gpu.thread_id z
// CHECK: gpu.block_id x
// CHECK: gpu.block_id y
// CHECK: gpu.block_id z
// CHECK: gpu.block_dim x
// CHECK: gpu.block_dim y
// CHECK: gpu.block_dim z
// CHECK: gpu.grid_dim x
// CHECK: gpu.grid_dim y
// CHECK: gpu.grid_dim z
// CHECK: %[[BIDX:.*]] = gpu.block_id x
// CHECK: %[[BDIMX:.*]] = gpu.block_dim x
// CHECK: %[[TIDX2:.*]] = gpu.thread_id x
// CHECK: %[[BLOCK_OFFSET:.*]] = arith.muli %[[BIDX]], %[[BDIMX]] : index
// CHECK: %[[GLOBAL_ID:.*]] = arith.addi %[[BLOCK_OFFSET]], %[[TIDX2]] : index
// CHECK: arith.index_cast %[[GLOBAL_ID]] : index to i64
// CHECK-NOT: forth.intrinsic

func.func private @main() {
  %0 = forth.stack !forth.stack
  %1 = forth.thread_id_x %0 : !forth.stack -> !forth.stack
  %2 = forth.thread_id_y %1 : !forth.stack -> !forth.stack
  %3 = forth.thread_id_z %2 : !forth.stack -> !forth.stack
  %4 = forth.block_id_x %3 : !forth.stack -> !forth.stack
  %5 = forth.block_id_y %4 : !forth.stack -> !forth.stack
  %6 = forth.block_id_z %5 : !forth.stack -> !forth.stack
  %7 = forth.block_dim_x %6 : !forth.stack -> !forth.stack
  %8 = forth.block_dim_y %7 : !forth.stack -> !forth.stack
  %9 = forth.block_dim_z %8 : !forth.stack -> !forth.stack
  %10 = forth.grid_dim_x %9 : !forth.stack -> !forth.stack
  %11 = forth.grid_dim_y %10 : !forth.stack -> !forth.stack
  %12 = forth.grid_dim_z %11 : !forth.stack -> !forth.stack
  %13 = forth.global_id %12 : !forth.stack -> !forth.stack
  return
}
