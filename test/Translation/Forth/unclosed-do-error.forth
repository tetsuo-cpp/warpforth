\ RUN: %not %warpforth-translate --forth-to-mlir %s 2>&1 | %FileCheck %s
\ CHECK: unclosed DO (expected LOOP or +LOOP)
\! kernel main
10 0 DO I
