\ RUN: %not %warpforth-translate --forth-to-mlir %s 2>&1 | %FileCheck %s
\ CHECK: {{.*}}else-without-if-error.forth:4:1: error: ELSE without matching IF
\! kernel main
ELSE
