\ RUN: %warpforth-translate --forth-to-mlir %s | %FileCheck %s

\ Verify BEGIN/WHILE/REPEAT generates condition check + body loop with cond_br

\ CHECK:       %[[S0:.*]] = forth.stack !forth.stack
\ CHECK-NEXT:  %[[S1:.*]] = forth.literal %[[S0]] 10 : !forth.stack -> !forth.stack
\ CHECK-NEXT:  cf.br ^bb1(%[[S1]] : !forth.stack)
\ CHECK:     ^bb1(%[[B1:.*]]: !forth.stack):
\ CHECK-NEXT:  %[[DUP:.*]] = forth.dup %[[B1]] : !forth.stack -> !forth.stack
\ CHECK-NEXT:  %[[L0:.*]] = forth.literal %[[DUP]] 0 : !forth.stack -> !forth.stack
\ CHECK-NEXT:  %[[GT:.*]] = forth.gt %[[L0]] : !forth.stack -> !forth.stack
\ CHECK-NEXT:  %[[PF:.*]], %[[FLAG:.*]] = forth.pop_flag %[[GT]] : !forth.stack -> !forth.stack, i1
\ CHECK-NEXT:  cf.cond_br %[[FLAG]], ^bb2(%[[PF]] : !forth.stack), ^bb3(%[[PF]] : !forth.stack)
\ CHECK:     ^bb2(%[[B2:.*]]: !forth.stack):
\ CHECK-NEXT:  %[[L1:.*]] = forth.literal %[[B2]] 1 : !forth.stack -> !forth.stack
\ CHECK-NEXT:  %[[SUB:.*]] = forth.sub %[[L1]] : !forth.stack -> !forth.stack
\ CHECK-NEXT:  cf.br ^bb1(%[[SUB]] : !forth.stack)
\ CHECK:     ^bb3(%[[B3:.*]]: !forth.stack):
\ CHECK-NEXT:  return
10 BEGIN DUP 0 > WHILE 1 - REPEAT
