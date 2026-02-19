\ RUN: %warpforth-translate --forth-to-mlir %s | %warpforth-opt --warpforth-pipeline | %FileCheck %s
\ RUN: %warpforth-translate --forth-to-mlir %s | %warpforth-opt --convert-forth-to-memref --convert-scf-to-cf --convert-forth-to-gpu | %FileCheck %s --check-prefix=MID

\ Verify nested DO/LOOP with I and J through full pipeline produces gpu.binary
\ CHECK: gpu.binary @warpforth_module

\ Verify intermediate MLIR: gpu.func with nested loop structure
\ MID: gpu.module @warpforth_module
\ MID: gpu.func @main(%arg0: memref<4xi64> {forth.param_name = "DATA"}) kernel
\ MID: cf.br
\ MID: arith.xori

\! kernel main
\! param DATA i64[4]
3 0 DO 4 0 DO J I + LOOP LOOP DATA 0 CELLS + !
