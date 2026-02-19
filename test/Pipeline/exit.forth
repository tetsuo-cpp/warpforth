\ RUN: %warpforth-translate --forth-to-mlir %s | %warpforth-opt --warpforth-pipeline | %FileCheck %s
\ CHECK: gpu.binary @warpforth_module

PARAM DATA 4
: DO-EXIT 1 IF EXIT THEN 42 ;
DO-EXIT DATA 0 CELLS + !
