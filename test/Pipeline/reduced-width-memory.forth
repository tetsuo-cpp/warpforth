\ RUN: %warpforth-translate --forth-to-mlir %s | %warpforth-opt --warpforth-pipeline | %FileCheck %s

\ Verify reduced-width memory ops compile through the full pipeline to gpu.binary
\ CHECK: gpu.binary @warpforth_module

\! kernel main
\! param DATA i64[256]
GLOBAL-ID CELLS DATA + I8@
GLOBAL-ID CELLS DATA + I8!
GLOBAL-ID CELLS DATA + I32@
GLOBAL-ID CELLS DATA + I32!
GLOBAL-ID CELLS DATA + HF@
GLOBAL-ID CELLS DATA + HF!
GLOBAL-ID CELLS DATA + F32@
GLOBAL-ID CELLS DATA + F32!
