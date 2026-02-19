\ RUN: %warpforth-translate --forth-to-mlir %s | %FileCheck %s

\ Verify +LOOP pops step from data stack and uses it as increment

\ CHECK:       %[[S0:.*]] = forth.stack !forth.stack
\ CHECK-NEXT:  %[[S1:.*]] = forth.literal %[[S0]] 10 : !forth.stack -> !forth.stack
\ CHECK-NEXT:  %[[S2:.*]] = forth.literal %[[S1]] 0 : !forth.stack -> !forth.stack
\ CHECK-NEXT:  %[[OS:.*]], %[[VAL:.*]] = forth.pop %[[S2]] : !forth.stack -> !forth.stack, i64
\ CHECK-NEXT:  %[[OS2:.*]], %[[LIM:.*]] = forth.pop %[[OS]] : !forth.stack -> !forth.stack, i64
\ CHECK-NEXT:  %[[ALLOCA:.*]] = memref.alloca() : memref<1xi64>
\ CHECK-NEXT:  %[[C0:.*]] = arith.constant 0 : index
\ CHECK-NEXT:  memref.store %[[VAL]], %[[ALLOCA]][%[[C0]]] : memref<1xi64>
\ CHECK-NEXT:  cf.br ^bb1(%[[OS2]] : !forth.stack)
\ CHECK:     ^bb1(%[[B1:.*]]: !forth.stack):
\ CHECK-NEXT:  %[[C0_2:.*]] = arith.constant 0 : index
\ CHECK-NEXT:  %[[LOAD1:.*]] = memref.load %[[ALLOCA]][%[[C0_2]]] : memref<1xi64>
\ CHECK-NEXT:  %[[CMP:.*]] = arith.cmpi slt, %[[LOAD1]], %[[LIM]] : i64
\ CHECK-NEXT:  cf.cond_br %[[CMP]], ^bb2(%[[B1]] : !forth.stack), ^bb3(%[[B1]] : !forth.stack)
\ CHECK:     ^bb2(%[[B2:.*]]: !forth.stack):
\ CHECK:       %[[STEP_S:.*]] = forth.literal %[[B2]] 2 : !forth.stack -> !forth.stack
\ CHECK-NEXT:  %[[POP_S:.*]], %[[STEP:.*]] = forth.pop %[[STEP_S]] : !forth.stack -> !forth.stack, i64
\ CHECK-NEXT:  %[[C0_3:.*]] = arith.constant 0 : index
\ CHECK-NEXT:  %[[LOAD2:.*]] = memref.load %[[ALLOCA]][%[[C0_3]]] : memref<1xi64>
\ CHECK-NEXT:  %[[ADDI:.*]] = arith.addi %[[LOAD2]], %[[STEP]] : i64
\ CHECK-NEXT:  memref.store %[[ADDI]], %[[ALLOCA]][%[[C0_3]]] : memref<1xi64>
\ CHECK-NEXT:  cf.br ^bb1(%[[POP_S]] : !forth.stack)
\ CHECK:     ^bb3(%[[B3:.*]]: !forth.stack):
\ CHECK-NEXT:  return
10 0 DO 2 +LOOP
