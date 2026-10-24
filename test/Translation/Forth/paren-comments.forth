\ Comment text (999) never reaches the IR.
\ RUN: %warpforth-translate --forth-to-mlir %s | %FileCheck %s --implicit-check-not=999

\ CHECK: func.func private @SQ(%arg0: !forth.stack) -> !forth.stack {
\ CHECK-NEXT: forth.dup
\ CHECK-NEXT: forth.muli
\ CHECK-NEXT: return

\ A word whose name merely starts with '(' is not a comment.
\ CHECK: func.func private @_28FOO_29(%arg0: !forth.stack) -> !forth.stack {
\ CHECK-NEXT: forth.constant %{{.*}}(7 : i64)

\ CHECK: func.func @main() attributes {forth.kernel}
\ CHECK: forth.constant %{{.*}}(3 : i64)
\ CHECK-NEXT: call @SQ
\ CHECK-NEXT: call @_28FOO_29
\ CHECK-NEXT: forth.constant %{{.*}}(4 : i64)
\ CHECK-NEXT: return
\! kernel main
: SQ ( n -- n*n ) DUP * ;
: (FOO) 7 ;
3 SQ ( 999 \ not a line comment
  999 spans lines ) (FOO)
(	999 tab-separated ) 4 ( 999 at end of input )
