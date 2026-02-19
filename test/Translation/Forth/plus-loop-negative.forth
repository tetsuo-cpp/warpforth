\ RUN: %warpforth-translate --forth-to-mlir %s | %FileCheck %s

\ Verify +LOOP with negative step uses crossing test (handles negative direction)

\ CHECK:       %[[S0:.*]] = forth.stack !forth.stack
\ CHECK-NEXT:  %[[S1:.*]] = forth.literal %[[S0]] 0 : !forth.stack -> !forth.stack
\ CHECK-NEXT:  %[[S2:.*]] = forth.literal %[[S1]] 10 : !forth.stack -> !forth.stack
\ CHECK-NEXT:  %[[OS:.*]], %[[VAL:.*]] = forth.pop %[[S2]] : !forth.stack -> !forth.stack, i64
\ CHECK-NEXT:  %[[OS2:.*]], %[[LIM:.*]] = forth.pop %[[OS]] : !forth.stack -> !forth.stack, i64
\ CHECK:       cf.br ^bb1(%[[OS2]] : !forth.stack)
\ CHECK:     ^bb1(%[[B1:.*]]: !forth.stack):
\ CHECK:       %[[STEP_S:.*]] = forth.literal %[[B1]] -1 : !forth.stack -> !forth.stack
\ CHECK-NEXT:  %[[POP_S:.*]], %[[STEP:.*]] = forth.pop %[[STEP_S]] : !forth.stack -> !forth.stack, i64
\ CHECK:       %[[OLD:.*]] = memref.load
\ CHECK:       %[[NEW:.*]] = arith.addi %[[OLD]], %[[STEP]] : i64
\ CHECK:       %[[D1:.*]] = arith.subi %[[OLD]], %[[LIM]] : i64
\ CHECK-NEXT:  %[[D2:.*]] = arith.subi %[[NEW]], %[[LIM]] : i64
\ CHECK-NEXT:  %[[XOR:.*]] = arith.xori %[[D1]], %[[D2]] : i64
\ CHECK-NEXT:  %[[ZERO:.*]] = arith.constant 0 : i64
\ CHECK-NEXT:  %[[CROSSED:.*]] = arith.cmpi slt, %[[XOR]], %[[ZERO]] : i64
\ CHECK-NEXT:  cf.cond_br %[[CROSSED]], ^bb2(%[[POP_S]] : !forth.stack), ^bb1(%[[POP_S]] : !forth.stack)
\ CHECK:     ^bb2(%{{.*}}: !forth.stack):
\ CHECK-NEXT:  return
0 10 DO -1 +LOOP
