\ RUN: %warpforth-translate --forth-to-mlir %s | %warpforth-opt --warpforth-pipeline | %FileCheck %s
\ RUN: %warpforth-translate --forth-to-mlir %s | %warpforth-opt --convert-forth-to-memref --convert-scf-to-cf --convert-forth-to-gpu | %FileCheck %s --check-prefix=MID

\ Verify that control flow through the full pipeline produces a gpu.binary
\ CHECK: gpu.binary @warpforth_module

\ Verify intermediate MLIR: gpu.func with conditional branching
\ MID: gpu.module @warpforth_module
\ MID: gpu.func @main(%arg0: memref<256xi32> {forth.param_name = "DATA"}) kernel
\ MID: memref.load
\ MID: arith.cmpi ne
\ MID: cf.cond_br
\ MID: gpu.return

\! kernel main
\! param DATA i32[256]
DATA @ 5 > IF DATA @ 1 + DATA ! THEN
