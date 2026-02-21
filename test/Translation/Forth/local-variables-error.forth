\ RUN: %not %warpforth-translate --forth-to-mlir %s 2>&1 | %FileCheck %s
\ CHECK: duplicate local variable name: X
\! kernel main
: BAD { x y x -- } x ;
BAD
