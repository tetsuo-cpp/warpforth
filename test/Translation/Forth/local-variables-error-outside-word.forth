\ RUN: %not %warpforth-translate --forth-to-mlir %s 2>&1 | %FileCheck %s
\ CHECK: local variables can only be declared inside a word definition
\! kernel main
{ x y -- }
