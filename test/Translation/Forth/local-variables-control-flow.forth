\ RUN: %warpforth-translate --forth-to-mlir %s | %FileCheck %s

\ Test that locals work across IF/ELSE/THEN control flow.
\ SSA values defined in the entry block dominate all subsequent blocks.

\ CHECK: func.func private @CLAMP(%arg0: !forth.stack) -> !forth.stack {
\ CHECK:   forth.pop
\ CHECK:   forth.pop
\ CHECK:   forth.pop
\ CHECK:   forth.push_value
\ CHECK:   forth.push_value
\ CHECK:   forth.push_value

\! kernel main
: CLAMP { val lo hi -- }
  val lo < IF lo ELSE
  val hi > IF hi ELSE
  val THEN THEN ;
0 10 5 CLAMP
