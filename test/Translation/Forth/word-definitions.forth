\ RUN: %warpforth-translate --forth-to-mlir %s | %FileCheck %s

\ CHECK: func.func private @DOUBLE(%arg0: !forth.stack) -> !forth.stack {
\ CHECK:   forth.dup
\ CHECK:   forth.addi
\ CHECK:   return %{{.*}} : !forth.stack
\ CHECK: }
\ CHECK: func.func private @main()
\ CHECK:   call @DOUBLE(%{{.*}}) : (!forth.stack) -> !forth.stack
\! kernel main
: DOUBLE DUP + ;
5 DOUBLE
