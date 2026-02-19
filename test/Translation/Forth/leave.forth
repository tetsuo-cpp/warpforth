\ RUN: %warpforth-translate --forth-to-mlir %s | %FileCheck %s

\ Verify LEAVE branches to the loop exit block.

\ CHECK:       %[[S0:.*]] = forth.stack !forth.stack
\ CHECK-NEXT:  %[[S1:.*]] = forth.literal %[[S0]] 10 : !forth.stack -> !forth.stack
\ CHECK-NEXT:  %[[S2:.*]] = forth.literal %[[S1]] 0 : !forth.stack -> !forth.stack
\ CHECK:       cf.br ^bb1(%{{.*}} : !forth.stack)
\ CHECK:     ^bb1(%[[B1:.*]]: !forth.stack):
\ CHECK-NEXT:  %[[TRUE:.*]] = arith.constant true
\ CHECK-NEXT:  cf.cond_br %[[TRUE]], ^bb[[EXIT:[0-9]+]](%[[B1]] : !forth.stack), ^bb{{[0-9]+}}(%[[B1]] : !forth.stack)
\ CHECK:     ^bb[[EXIT]](%[[B3:.*]]: !forth.stack):
\ CHECK-NEXT:  return

10 0 DO LEAVE LOOP
