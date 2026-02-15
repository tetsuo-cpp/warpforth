\ RUN: %warpforth-translate --forth-to-mlir %s | %FileCheck %s

\ Hyphen becomes underscore
\ CHECK: func.func private @MY_WORD

\ Underscore becomes double underscore
\ CHECK: func.func private @UNDER__SCORE

\ Leading digit gets underscore prefix
\ CHECK: func.func private @_2START

: MY-WORD 1 ;
: UNDER_SCORE 2 ;
: 2START 3 ;
