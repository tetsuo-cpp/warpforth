\ RUN: %warpforth-translate --forth-to-mlir %s | %warpforth-opt --warpforth-pipeline | %FileCheck %s
\ RUN: %warpforth-translate --forth-to-mlir %s | %warpforth-opt --convert-forth-to-memref --canonicalize | %FileCheck %s --check-prefix=MID --implicit-check-not="arith.constant true"
\ CHECK: gpu.binary @warpforth_module
\ MID-LABEL: func.func private @DO_EXIT
\ MID-COUNT-2: return

\! kernel main
\! param DATA i64[4]
: DO-EXIT 1 IF EXIT THEN 42 ;
DO-EXIT DATA 0 CELLS + !
