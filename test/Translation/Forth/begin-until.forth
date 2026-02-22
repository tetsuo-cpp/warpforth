\ RUN: %warpforth-translate --forth-to-mlir %s | %FileCheck %s

\ Verify BEGIN/UNTIL generates loop with pop_flag + cond_br

\ CHECK:       %[[S0:.*]] = forth.stack !forth.stack
\ CHECK-NEXT:  %[[S1:.*]] = forth.constant %[[S0]](10 : i32) : !forth.stack -> !forth.stack
\ CHECK-NEXT:  cf.br ^bb1(%[[S1]] : !forth.stack)
\ CHECK:     ^bb1(%[[B1:.*]]: !forth.stack):
\ CHECK-NEXT:  %[[L1:.*]] = forth.constant %[[B1]](1 : i32) : !forth.stack -> !forth.stack
\ CHECK-NEXT:  %[[SUB:.*]] = forth.subi %[[L1]] : !forth.stack -> !forth.stack
\ CHECK-NEXT:  %[[DUP:.*]] = forth.dup %[[SUB]] : !forth.stack -> !forth.stack
\ CHECK-NEXT:  %[[ZEQ:.*]] = forth.zero_eq %[[DUP]] : !forth.stack -> !forth.stack
\ CHECK-NEXT:  %[[PF:.*]], %[[FLAG:.*]] = forth.pop_flag %[[ZEQ]] : !forth.stack -> !forth.stack, i1
\ CHECK-NEXT:  cf.cond_br %[[FLAG]], ^bb2(%[[PF]] : !forth.stack), ^bb1(%[[PF]] : !forth.stack)
\ CHECK:     ^bb2(%[[B2:.*]]: !forth.stack):
\ CHECK-NEXT:  return
\! kernel main
10 BEGIN 1 - DUP 0= UNTIL
