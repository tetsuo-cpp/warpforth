\ RUN: %warpforth-translate --forth-to-mlir %s | %FileCheck %s

\ Hyphen becomes underscore
\ CHECK: func.func private @my_word

\ Underscore becomes double underscore
\ CHECK: func.func private @under__score

\ Leading digit gets underscore prefix
\ CHECK: func.func private @_2start

: my-word 1 ;
: under_score 2 ;
: 2start 3 ;
