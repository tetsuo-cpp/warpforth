// RUN: %warpforth-opt --convert-forth-to-gpu %s | %FileCheck %s

// CHECK: gpu.module @warpforth_module
// CHECK: gpu.func @main() kernel

// CHECK: gpu.thread_id x
// CHECK: gpu.thread_id y
// CHECK: gpu.block_id z
// CHECK: gpu.block_dim x
// CHECK: gpu.grid_dim y
// CHECK: gpu.barrier

module {
  func.func private @main() attributes {forth.kernel} {
    %0 = gpu.thread_id x
    %1 = gpu.thread_id y
    %2 = gpu.block_id z
    %3 = gpu.block_dim x
    %4 = gpu.grid_dim y
    gpu.barrier
    return
  }
}
