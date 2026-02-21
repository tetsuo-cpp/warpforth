\ RUN: %not %warpforth-translate --forth-to-mlir %s 2>&1 | %FileCheck %s
\ CHECK: REPEAT without matching WHILE
\! kernel main
REPEAT
