\ RUN: %warpforth-translate --forth-to-mlir %s | %warpforth-opt --warpforth-pipeline | %FileCheck %s

\ Verify that local variables compile through the full pipeline to gpu.binary.
\ CHECK: gpu.binary @warpforth_module

\! kernel main
\! param DATA i32[256]
: ADD3 { a b c -- } a b + c + ;
1 2 3 ADD3
GLOBAL-ID CELLS DATA + !
