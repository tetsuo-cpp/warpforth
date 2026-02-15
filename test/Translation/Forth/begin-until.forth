\ RUN: %warpforth-translate --forth-to-mlir %s | %FileCheck %s

\ Verify BEGIN/UNTIL parsing produces forth.begin_until with body region

\ CHECK: %[[S0:.*]] = forth.stack
\ CHECK: %[[S1:.*]] = forth.literal %[[S0]] 10
\ CHECK: %[[LOOP:.*]] = forth.begin_until %[[S1]]
\ CHECK:   ^bb0(%[[ARG:.*]]: !forth.stack):
\ CHECK:   forth.literal %[[ARG]] 1
\ CHECK:   forth.sub
\ CHECK:   forth.dup
\ CHECK:   forth.zero_eq
\ CHECK:   forth.yield
\ CHECK: }
10 BEGIN 1 - DUP 0= UNTIL
