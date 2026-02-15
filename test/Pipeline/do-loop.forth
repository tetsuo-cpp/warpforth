\ RUN: %warpforth-translate --forth-to-mlir %s | %warpforth-opt --warpforth-pipeline | %FileCheck %s
\ RUN: %warpforth-translate --forth-to-mlir %s | %warpforth-opt --convert-forth-to-memref --convert-scf-to-cf --convert-forth-to-gpu | %FileCheck %s --check-prefix=MID

\ Verify that DO/LOOP through the full pipeline produces a gpu.binary
\ CHECK: gpu.binary @warpforth_module

\ Verify intermediate MLIR: gpu.func with loop structure
\ MID: gpu.module @warpforth_module
\ MID: gpu.func @main(%arg0: memref<4xi64> {forth.param_name = "data"}) kernel
\ MID: cf.br
\ MID: cf.cond_br
\ MID: gpu.return

param data 4
10 0 DO I LOOP data 0 cells + !
