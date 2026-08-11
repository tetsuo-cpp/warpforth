\ RUN: %not %warpforth-translate --forth-to-mlir %s 2>&1 | %FileCheck %s
\ CHECK: unclosed DO (expected LOOP or +LOOP)
\ CHECK-NOT: reference to block defined in another region
\! kernel main
: FIRST 10 0 DO I ;
: SECOND LOOP ;
