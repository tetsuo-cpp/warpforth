\ RUN: %not %warpforth-translate --forth-to-mlir %s 2>&1 | %FileCheck %s
\ CHECK: {{.*}}leave-without-do-error.forth:4:1: error: LEAVE without matching DO
\! kernel main
LEAVE
