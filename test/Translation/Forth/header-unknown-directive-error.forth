\ RUN: %not %warpforth-translate --forth-to-mlir %s 2>&1 | %FileCheck %s
\ CHECK: unknown header directive: BOGUS
\! kernel main
\! bogus foo bar
