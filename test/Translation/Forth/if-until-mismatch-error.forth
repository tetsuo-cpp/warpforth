\ RUN: %not %warpforth-translate --forth-to-mlir %s 2>&1 | %FileCheck %s
\ CHECK: UNTIL without matching BEGIN
\! kernel main
IF UNTIL
