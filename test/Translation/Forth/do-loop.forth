\ RUN: %warpforth-translate --forth-to-mlir %s | %FileCheck %s

\ Verify DO/LOOP parsing produces forth.do_loop with forth.loop_index

\ CHECK: %[[S0:.*]] = forth.stack
\ CHECK: %[[S1:.*]] = forth.literal %[[S0]] 10
\ CHECK: %[[S2:.*]] = forth.literal %[[S1]] 0
\ CHECK: %[[LOOP:.*]] = forth.do_loop %[[S2]]
\ CHECK:   ^bb0(%[[ARG:.*]]: !forth.stack):
\ CHECK:   forth.loop_index %[[ARG]]
\ CHECK:   forth.yield
\ CHECK: }
10 0 DO I LOOP
