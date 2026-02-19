\ RUN: %warpforth-translate --forth-to-mlir %s | %warpforth-opt --warpforth-pipeline | %FileCheck %s
\ CHECK: gpu.binary @warpforth_module

\! kernel main
\! param DATA i64[4]
: FIND-FIVE  10 0 DO I 5 = IF UNLOOP EXIT THEN LOOP 0 ;
FIND-FIVE DATA 0 CELLS + !
