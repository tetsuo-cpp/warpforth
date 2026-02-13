\ RUN: %warpforth-translate --forth-to-mlir %s | %FileCheck %s

\ CHECK: func.func private @double(%arg0: !forth.stack) -> !forth.stack {
\ CHECK:   forth.dup
\ CHECK:   forth.add
\ CHECK:   return %{{.*}} : !forth.stack
\ CHECK: }
\ CHECK: func.func private @main()
\ CHECK:   call @double(%{{.*}}) : (!forth.stack) -> !forth.stack
: double dup + ;
5 double
