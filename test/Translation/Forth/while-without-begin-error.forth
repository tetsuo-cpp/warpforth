\ RUN: %not %warpforth-translate --forth-to-mlir %s 2>&1 | %FileCheck %s
\ CHECK: {{.*}}while-without-begin-error.forth:4:1: error: WHILE without matching BEGIN
\! kernel main
WHILE
