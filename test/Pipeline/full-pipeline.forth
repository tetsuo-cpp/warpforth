\ RUN: %warpforth-translate --forth-to-mlir %s | %warpforth-opt --warpforth-pipeline | %FileCheck %s
\ RUN: %warpforth-translate --forth-to-mlir %s | %warpforth-opt --convert-forth-to-memref --convert-forth-to-gpu | %FileCheck %s --check-prefix=MID

\ Verify that Forth source through the full pipeline produces a gpu.binary
\ CHECK: gpu.binary @warpforth_module

\ Verify intermediate MLIR structure at the memref+gpu stage
\ MID: gpu.module @warpforth_module
\ MID: gpu.func @main(%arg0: memref<256xi64> {forth.param_name = "DATA"}) kernel
\ MID: memref.alloca() : memref<256xi64>
\ MID: gpu.thread_id  x
\ MID: memref.extract_aligned_pointer_as_index %arg0
\ MID: llvm.load
\ MID: llvm.store
\ MID: gpu.return

PARAM DATA 256
GLOBAL-ID CELLS DATA + @
1 +
GLOBAL-ID CELLS DATA + !
