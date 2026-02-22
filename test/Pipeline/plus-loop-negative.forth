\ RUN: %warpforth-translate --forth-to-mlir %s | %warpforth-opt --warpforth-pipeline | %FileCheck %s

\ Verify that +LOOP with negative step through the full pipeline produces a gpu.binary
\ CHECK: gpu.binary @warpforth_module

\! kernel main
\! param DATA i32[4]
0 10 DO I DATA 0 CELLS + ! -1 +LOOP
