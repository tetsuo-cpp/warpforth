\ RUN: %warpforth-translate --forth-to-mlir %s | %warpforth-opt --warpforth-pipeline | %FileCheck %s

\ Verify that +LOOP through the full pipeline produces a gpu.binary
\ CHECK: gpu.binary @warpforth_module

\! kernel main
\! param DATA i64[4]
10 0 DO I DATA 0 CELLS + ! 2 +LOOP
