\ RUN: %warpforth-translate --forth-to-mlir %s | %warpforth-opt --warpforth-pipeline | %FileCheck %s
\ RUN: %warpforth-translate --forth-to-mlir %s | %warpforth-opt --lower-forth-stack --convert-forth-to-gpu | %FileCheck %s --check-prefix=MID

\ Verify that Forth source through the full pipeline produces a gpu.binary
\ CHECK: gpu.binary @warpforth_module

\ Verify intermediate MLIR after stack lowering and GPU conversion. The kernel
\ is straight-line, so the loaded cell reaches the store as an SSA value with
\ no stack buffer in between.
\ MID: gpu.module @warpforth_module
\ MID: gpu.func @main(%arg0: memref<256xi64> {forth.param_name = "DATA"}) kernel
\ MID-NOT: memref.alloca
\ MID: gpu.thread_id  x
\ MID: memref.extract_aligned_pointer_as_index %arg0
\ MID: %[[LOADED:.*]] = llvm.load %{{.*}} : !llvm.ptr<1> -> i64
\ MID: %[[INCREMENTED:.*]] = arith.addi %[[LOADED]], %{{.*}} : i64
\ MID: llvm.store %[[INCREMENTED]], %{{.*}} : i64, !llvm.ptr<1>
\ MID: gpu.return

\! kernel main
\! param DATA i64[256]
GLOBAL-ID CELLS DATA + @
1 +
GLOBAL-ID CELLS DATA + !
