\ RUN: %warpforth-translate --forth-to-mlir %s | %FileCheck %s

\ Verify BEGIN/WHILE/REPEAT parsing produces forth.begin_while_repeat
\ with condition and body regions

\ CHECK: %[[S0:.*]] = forth.stack
\ CHECK: %[[S1:.*]] = forth.literal %[[S0]] 10
\ CHECK: %[[LOOP:.*]] = forth.begin_while_repeat %[[S1]]
\ CHECK:   ^bb0(%[[CARG:.*]]: !forth.stack):
\ CHECK:   forth.dup
\ CHECK:   forth.literal
\ CHECK:   forth.gt
\ CHECK:   forth.yield %{{.*}} while_cond
\ CHECK: } do {
\ CHECK:   ^bb0(%[[BARG:.*]]: !forth.stack):
\ CHECK:   forth.literal
\ CHECK:   forth.sub
\ CHECK:   forth.yield
\ CHECK: }
10 BEGIN DUP 0 > WHILE 1 - REPEAT
