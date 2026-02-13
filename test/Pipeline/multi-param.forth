\ RUN: %warpforth-translate --forth-to-mlir %s | %warpforth-opt --warpforth-pipeline | %FileCheck %s
\ RUN: %warpforth-translate --forth-to-mlir %s | %warpforth-opt --convert-forth-to-memref --convert-forth-to-gpu | %FileCheck %s --check-prefix=MID

\ Verify multiple params survive the full pipeline
\ CHECK: gpu.binary @warpforth_module

\ Verify both params appear in kernel signature and are used correctly
\ MID: gpu.func @main(%arg0: memref<256xi64> {forth.param_name = "input"}, %arg1: memref<256xi64> {forth.param_name = "output"}) kernel
\ MID: memref.extract_aligned_pointer_as_index %arg0
\ MID: llvm.load
\ MID: memref.extract_aligned_pointer_as_index %arg1
\ MID: llvm.store
\ MID: gpu.return

param input 256
param output 256
global-id cells input + @
2 *
global-id cells output + !
