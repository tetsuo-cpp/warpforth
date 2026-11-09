\ RUN: %warpforth-translate --forth-to-mlir %s | %warpforth-opt --warpforth-pipeline | %FileCheck %s
\ RUN: %warpforth-translate --forth-to-mlir %s | %warpforth-opt --lower-forth-stack --convert-forth-to-gpu | %FileCheck %s --check-prefix=MID

\ Verify that Forth with f64 params through the full pipeline produces a gpu.binary
\ CHECK: gpu.binary @warpforth_module

\ Verify intermediate MLIR after stack lowering and GPU conversion. The f64
\ cells travel as i64 bits, and no stack buffer holds them.
\ MID: gpu.module @warpforth_module
\ MID: gpu.func @main(%arg0: memref<256xf64> {forth.param_name = "DATA"}, %arg1: f64 {forth.param_name = "SCALE"}) kernel
\ MID-NOT: memref.alloca
\ MID: memref.extract_aligned_pointer_as_index %arg0
\ MID: llvm.load %{{.*}} : !llvm.ptr<1> -> f64
\ MID: arith.bitcast %arg1 : f64 to i64
\ MID: arith.mulf %{{.*}}, %{{.*}} : f64
\ MID: llvm.store %{{.*}}, %{{.*}} : f64, !llvm.ptr<1>
\ MID: gpu.return

\! kernel main
\! param DATA f64[256]
\! param SCALE f64
GLOBAL-ID CELLS DATA + F@
SCALE F*
GLOBAL-ID CELLS DATA + F!
