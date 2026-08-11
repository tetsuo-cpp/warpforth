\ RUN: %not %warpforth-translate --forth-to-mlir %s 2>&1 | %FileCheck %s
\ CHECK: {{.*}}repeat-without-while-error.forth:4:1: error: REPEAT without matching WHILE
\! kernel main
REPEAT
