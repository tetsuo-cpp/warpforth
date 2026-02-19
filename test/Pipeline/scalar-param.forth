\ RUN: %warpforth-translate --forth-to-mlir %s | %warpforth-opt --warpforth-pipeline | %FileCheck %s
\ RUN: %warpforth-translate --forth-to-mlir %s | %warpforth-opt --convert-forth-to-memref --convert-forth-to-gpu | %FileCheck %s --check-prefix=MID

\ Verify mixed scalar + array params survive the full pipeline
\ CHECK: gpu.binary @warpforth_module

\ Verify scalar becomes i64 arg, array becomes memref
\ MID: gpu.func @main(
\ MID-SAME: i64 {forth.param_name = "SCALE"}
\ MID-SAME: memref<256xi64> {forth.param_name = "DATA"}
\ MID-SAME: kernel
\ MID: gpu.return

\! kernel main
\! param SCALE i64
\! param DATA i64[256]
GLOBAL-ID
DUP CELLS DATA + @
SCALE *
SWAP CELLS DATA + !
