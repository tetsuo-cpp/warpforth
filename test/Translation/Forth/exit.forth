\ RUN: %warpforth-translate --forth-to-mlir %s | %FileCheck %s

\ CHECK: func.func private @EARLY_EXIT(%[[A:.*]]: !forth.stack) -> !forth.stack
\ CHECK:   cf.cond_br %{{.*}}, ^[[THEN:bb.*]](%{{.*}}), ^[[JOIN:bb.*]](%{{.*}})
\ CHECK: ^[[THEN]](%[[T:.*]]: !forth.stack):
\ CHECK:   cf.cond_br %true, ^[[RET:bb.*]](%[[T]]{{.*}}), ^[[DEAD:bb.*]](%[[T]]
\ CHECK: ^[[JOIN]](%{{.*}}: !forth.stack):
\ CHECK:   return %{{.*}} : !forth.stack
\ CHECK: ^[[RET]](%[[R:.*]]: !forth.stack):
\ CHECK:   return %[[R]] : !forth.stack

: EARLY-EXIT 1 IF EXIT THEN 42 ;
EARLY-EXIT
