\ RUN: %warpforth-translate --forth-to-mlir %s | %FileCheck %s

\ Verify IF/ELSE/THEN generates pop_flag + cond_br control flow

\ Basic IF/ELSE/THEN
\ CHECK:       %[[S0:.*]] = forth.stack !forth.stack
\ CHECK-NEXT:  %[[S1:.*]] = forth.constant %[[S0]](1 : i32) : !forth.stack -> !forth.stack
\ CHECK-NEXT:  %[[PF1:.*]], %[[FLAG1:.*]] = forth.pop_flag %[[S1]] : !forth.stack -> !forth.stack, i1
\ CHECK-NEXT:  cf.cond_br %[[FLAG1]], ^bb1(%[[PF1]] : !forth.stack), ^bb2(%[[PF1]] : !forth.stack)
\ CHECK:     ^bb1(%[[B1:.*]]: !forth.stack):
\ CHECK-NEXT:  %[[L42:.*]] = forth.constant %[[B1]](42 : i32) : !forth.stack -> !forth.stack
\ CHECK-NEXT:  cf.br ^bb3(%[[L42]] : !forth.stack)
\ CHECK:     ^bb2(%[[B2:.*]]: !forth.stack):
\ CHECK-NEXT:  %[[L99:.*]] = forth.constant %[[B2]](99 : i32) : !forth.stack -> !forth.stack
\ CHECK-NEXT:  cf.br ^bb3(%[[L99]] : !forth.stack)
\! kernel main
1 IF 42 ELSE 99 THEN

\ Basic IF/THEN (no ELSE - fallthrough on false)
\ CHECK:     ^bb3(%[[B3:.*]]: !forth.stack):
\ CHECK-NEXT:  %[[S2:.*]] = forth.constant %[[B3]](0 : i32) : !forth.stack -> !forth.stack
\ CHECK-NEXT:  %[[PF2:.*]], %[[FLAG2:.*]] = forth.pop_flag %[[S2]] : !forth.stack -> !forth.stack, i1
\ CHECK-NEXT:  cf.cond_br %[[FLAG2]], ^bb4(%[[PF2]] : !forth.stack), ^bb5(%[[PF2]] : !forth.stack)
\ CHECK:     ^bb4(%[[B4:.*]]: !forth.stack):
\ CHECK-NEXT:  %[[L7:.*]] = forth.constant %[[B4]](7 : i32) : !forth.stack -> !forth.stack
\ CHECK-NEXT:  cf.br ^bb5(%[[L7]] : !forth.stack)
\ CHECK:     ^bb5(%[[B5:.*]]: !forth.stack):
\ CHECK-NEXT:  return
0 IF 7 THEN
