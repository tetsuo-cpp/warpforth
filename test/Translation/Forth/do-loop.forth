\ RUN: %warpforth-translate --forth-to-mlir %s | %FileCheck %s

\ Verify DO/LOOP generates post-test loop with crossing test

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
\ CHECK-NEXT:  %[[PUSH:.*]] = forth.push_value %[[B1]], %[[LOAD1]] : !forth.stack, i64 -> !forth.stack
\ CHECK-NEXT:  %[[C1:.*]] = arith.constant 1 : i64
\ CHECK-NEXT:  %[[C0_3:.*]] = arith.constant 0 : index
\ CHECK-NEXT:  %[[OLD:.*]] = memref.load %[[ALLOCA]][%[[C0_3]]] : memref<1xi64>
\ CHECK-NEXT:  %[[NEW:.*]] = arith.addi %[[OLD]], %[[C1]] : i64
\ CHECK-NEXT:  memref.store %[[NEW]], %[[ALLOCA]][%[[C0_3]]] : memref<1xi64>
\ CHECK-NEXT:  %[[D1:.*]] = arith.subi %[[OLD]], %[[LIM]] : i64
\ CHECK-NEXT:  %[[D2:.*]] = arith.subi %[[NEW]], %[[LIM]] : i64
\ CHECK-NEXT:  %[[XOR:.*]] = arith.xori %[[D1]], %[[D2]] : i64
\ CHECK-NEXT:  %[[ZERO:.*]] = arith.constant 0 : i64
\ CHECK-NEXT:  %[[CROSSED:.*]] = arith.cmpi slt, %[[XOR]], %[[ZERO]] : i64
\ CHECK-NEXT:  cf.cond_br %[[CROSSED]], ^bb2(%[[PUSH]] : !forth.stack), ^bb1(%[[PUSH]] : !forth.stack)
\ CHECK:     ^bb2(%[[B2:.*]]: !forth.stack):
\ CHECK-NEXT:  return
\! kernel main
10 0 DO I LOOP
