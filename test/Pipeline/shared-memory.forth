\ RUN: %warpforth-translate --forth-to-mlir %s | %warpforth-opt --warpforth-pipeline | %FileCheck %s
\ RUN: %warpforth-translate --forth-to-mlir %s | %warpforth-opt --convert-forth-to-memref --convert-forth-to-gpu | %FileCheck %s --check-prefix=MID

\ Verify that shared memory through the full pipeline produces a gpu.binary
\ CHECK: gpu.binary @warpforth_module

\ Verify intermediate MLIR structure: shared alloca becomes workgroup attribution
\ MID: gpu.module @warpforth_module
\ MID: gpu.func @main(%arg0: memref<256xi64> {forth.param_name = "DATA"})
\ MID-SAME: workgroup(%{{.*}}: memref<256xi64, #gpu.address_space<workgroup>>)
\ MID-SAME: kernel
\ MID: memref.extract_aligned_pointer_as_index %{{.*}} : memref<256xi64, #gpu.address_space<workgroup>>
\ MID: llvm.store
\ MID: gpu.return

\! kernel main
\! param DATA i64[256]
\! shared SCRATCH i64[256]
GLOBAL-ID CELLS SCRATCH + !
GLOBAL-ID CELLS SCRATCH + @
GLOBAL-ID CELLS DATA + !
