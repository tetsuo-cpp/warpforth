\ RUN: %warpforth-translate --forth-to-mlir %s | %warpforth-opt --warpforth-pipeline | %FileCheck %s
\ RUN: %warpforth-translate --forth-to-mlir %s | %warpforth-opt --convert-forth-to-memref --convert-forth-to-gpu | %FileCheck %s --check-prefix=MID
\ CHECK: gpu.binary @warpforth_module
\ MID: gpu.func @main
\ MID: gpu.barrier
\ MID: gpu.return
\! kernel main
\! param DATA i64[256]
\! shared SCRATCH i64[256]
GLOBAL-ID CELLS SCRATCH + @ BARRIER GLOBAL-ID CELLS DATA + !
