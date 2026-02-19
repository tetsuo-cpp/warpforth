\ RUN: %not %warpforth-translate --forth-to-mlir %s 2>&1 | %FileCheck %s
\ CHECK: 'J' requires 2 nested DO/LOOP(s)
\! kernel main
10 0 DO J LOOP
