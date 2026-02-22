\ RUN: %warpforth-translate --forth-to-mlir %s | %warpforth-opt --warpforth-pipeline | %FileCheck %s
\ RUN: %warpforth-translate --forth-to-mlir %s | %warpforth-opt --convert-forth-to-memref --convert-forth-to-gpu | %FileCheck %s --check-prefix=MID

\ Verify mixed scalar + array params survive the full pipeline
\ CHECK: gpu.binary @warpforth_module

\ Verify scalar becomes i32 arg, array becomes memref
\ MID: gpu.func @main(
\ MID-SAME: i32 {forth.param_name = "SCALE"}
\ MID-SAME: memref<256xi32> {forth.param_name = "DATA"}
\ MID-SAME: kernel
\ MID: gpu.return

\! kernel main
\! param SCALE i32
\! param DATA i32[256]
GLOBAL-ID
DUP CELLS DATA + @
SCALE *
SWAP CELLS DATA + !
