\ RUN: %warpforth-translate --forth-to-mlir %s | %warpforth-opt --warpforth-pipeline | %FileCheck %s

\ Verify narrow memory operations through full pipeline produce a gpu.binary
\ CHECK: gpu.binary @warpforth_module
\! kernel main
\! param DATA i32[256]
GLOBAL-ID CELLS DATA + HF@
GLOBAL-ID CELLS DATA + HF!
GLOBAL-ID CELLS DATA + BF@
GLOBAL-ID CELLS DATA + BF!
GLOBAL-ID CELLS DATA + I8@
GLOBAL-ID CELLS DATA + I8!
GLOBAL-ID CELLS DATA + I16@
GLOBAL-ID CELLS DATA + I16!
