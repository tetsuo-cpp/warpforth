\ RUN: %warpforth-translate --forth-to-mlir %s | %FileCheck %s

\ Hex, nan and inf spellings are ordinary words, not float literals.
\ CHECK-NOT: f64
\ CHECK: func.func private @NAN(
\ CHECK: func.func private @INF(
\ CHECK: func.func private @INFINITY(
\ CHECK: func.func private @_0X1P3(
\ CHECK: func.func @main()
\ CHECK: call @NAN(
\ CHECK: call @INF(
\ CHECK: call @INFINITY(
\ CHECK: call @_0X1P3(
\! kernel main
: nan 7 ;
: inf 8 ;
: infinity 9 ;
: 0x1p3 10 ;
nan inf infinity 0x1p3
