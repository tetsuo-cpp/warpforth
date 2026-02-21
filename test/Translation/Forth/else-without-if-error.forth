\ RUN: %not %warpforth-translate --forth-to-mlir %s 2>&1 | %FileCheck %s
\ CHECK: ELSE without matching IF
\! kernel main
ELSE
