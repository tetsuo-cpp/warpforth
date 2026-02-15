\ RUN: %warpforth-translate --forth-to-mlir %s | %warpforth-opt --warpforth-pipeline | %FileCheck %s
\ RUN: %warpforth-translate --forth-to-mlir %s | %warpforth-opt --convert-forth-to-memref --convert-scf-to-cf --convert-forth-to-gpu | %FileCheck %s --check-prefix=MID

\ Verify that BEGIN/UNTIL through the full pipeline produces a gpu.binary
\ CHECK: gpu.binary @warpforth_module

\ Verify intermediate MLIR: gpu.func with loop back-edge and conditional branch
\ MID: gpu.module @warpforth_module
\ MID: gpu.func @main(%arg0: memref<4xi64> {forth.param_name = "DATA"}) kernel
\ MID: cf.br
\ MID: cf.cond_br
\ MID: gpu.return

PARAM DATA 4
10 BEGIN 1 - DUP 0= UNTIL DATA 0 CELLS + !
