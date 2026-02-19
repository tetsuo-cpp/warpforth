\ RUN: %warpforth-translate --forth-to-mlir %s | %FileCheck %s

\ Verify the UNLOOP EXIT idiom: early return from a DO LOOP inside a word.

\ CHECK: func.func private @FIND_FIVE(%{{.*}}: !forth.stack) -> !forth.stack
\ CHECK: memref.alloca
\ CHECK: cf.br ^bb[[#BODY:]]
\ CHECK: ^bb[[#BODY]](%{{.*}}: !forth.stack):
\ CHECK: forth.eq
\ CHECK: cf.cond_br %{{.*}}, ^bb[[#THEN:]](%{{.*}}), ^bb[[#ENDIF:]](%{{.*}})
\ CHECK: ^bb[[#EXIT:]](%{{.*}}: !forth.stack):
\ CHECK: return
\ CHECK: ^bb[[#THEN]](%[[T:.*]]: !forth.stack):
\ CHECK: cf.cond_br %true, ^bb[[#RET:]](%[[T]]{{.*}})
\ CHECK: ^bb[[#RET]](%[[R:.*]]: !forth.stack):
\ CHECK: return %[[R]] : !forth.stack

: FIND-FIVE  10 0 DO I 5 = IF UNLOOP EXIT THEN LOOP 0 ;
FIND-FIVE
