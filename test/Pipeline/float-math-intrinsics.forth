\ RUN: %warpforth-translate --forth-to-mlir %s | %warpforth-opt --warpforth-pipeline | %FileCheck %s

\ Verify float math intrinsics lower through the full pipeline to gpu.binary
\ CHECK: gpu.binary @warpforth_module

\! kernel main
\! param data f32[256]
GLOBAL-ID CELLS data + F@
FABS FEXP FSQRT FLOG FNEG
GLOBAL-ID CELLS data + F@
FMAX FMIN
GLOBAL-ID CELLS data + F!
