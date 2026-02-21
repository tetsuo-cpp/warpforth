\ RUN: %not %warpforth-translate --forth-to-mlir %s 2>&1 | %FileCheck %s
\ CHECK: WHILE without matching BEGIN
\! kernel main
WHILE
