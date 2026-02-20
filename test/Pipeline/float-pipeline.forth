\ RUN: %warpforth-translate --forth-to-mlir %s | %warpforth-opt --warpforth-pipeline | %FileCheck %s
\ RUN: %warpforth-translate --forth-to-mlir %s | %warpforth-opt --convert-forth-to-memref --convert-forth-to-gpu | %FileCheck %s --check-prefix=MID

\ Verify that Forth with f64 params through the full pipeline produces a gpu.binary
\ CHECK: gpu.binary @warpforth_module

\ Verify intermediate MLIR structure at the memref+gpu stage
\ MID: gpu.module @warpforth_module
\ MID: gpu.func @main(%arg0: memref<256xf64> {forth.param_name = "DATA"}, %arg1: f64 {forth.param_name = "SCALE"}) kernel
\ MID: memref.alloca() : memref<256xi64>
\ MID: memref.extract_aligned_pointer_as_index %arg0
\ MID: arith.bitcast %{{.*}} : f64 to i64
\ MID: gpu.return

\! kernel main
\! param DATA f64[256]
\! param SCALE f64
GLOBAL-ID CELLS DATA + F@
SCALE F*
GLOBAL-ID CELLS DATA + F!
