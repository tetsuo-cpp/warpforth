\ RUN: %warpforth-translate --forth-to-mlir %s | %FileCheck %s
\ CHECK: forth.barrier
\! kernel main
\! param DATA i32[256]
GLOBAL-ID CELLS DATA + @ BARRIER DROP
