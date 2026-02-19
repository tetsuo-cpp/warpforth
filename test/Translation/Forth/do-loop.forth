\ RUN: %warpforth-translate --forth-to-mlir %s | %FileCheck %s

\ Verify DO/LOOP generates loop counter with memref.alloca, pop, cmpi, cond_br

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
\ CHECK-NEXT:  %[[C0_3:.*]] = arith.constant 0 : index
\ CHECK-NEXT:  %[[LOAD2:.*]] = memref.load %[[ALLOCA]][%[[C0_3]]] : memref<1xi64>
\ CHECK-NEXT:  %[[PUSH:.*]] = forth.push_value %[[B2]], %[[LOAD2]] : !forth.stack, i64 -> !forth.stack
\ CHECK-NEXT:  %[[C0_4:.*]] = arith.constant 0 : index
\ CHECK-NEXT:  %[[LOAD3:.*]] = memref.load %[[ALLOCA]][%[[C0_4]]] : memref<1xi64>
\ CHECK-NEXT:  %[[C1:.*]] = arith.constant 1 : i64
\ CHECK-NEXT:  %[[ADDI:.*]] = arith.addi %[[LOAD3]], %[[C1]] : i64
\ CHECK-NEXT:  memref.store %[[ADDI]], %[[ALLOCA]][%[[C0_4]]] : memref<1xi64>
\ CHECK-NEXT:  cf.br ^bb1(%[[PUSH]] : !forth.stack)
\ CHECK:     ^bb3(%[[B3:.*]]: !forth.stack):
\ CHECK-NEXT:  return
10 0 DO I LOOP
