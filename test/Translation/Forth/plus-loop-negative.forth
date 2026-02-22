\ RUN: %warpforth-translate --forth-to-mlir %s | %FileCheck %s

\ Verify +LOOP with negative step uses crossing test (handles negative direction)

\ CHECK:       %[[S0:.*]] = forth.stack !forth.stack
\ CHECK-NEXT:  %[[S1:.*]] = forth.constant %[[S0]](0 : i32) : !forth.stack -> !forth.stack
\ CHECK-NEXT:  %[[S2:.*]] = forth.constant %[[S1]](10 : i32) : !forth.stack -> !forth.stack
\ CHECK-NEXT:  %[[OS:.*]], %[[VAL:.*]] = forth.pop %[[S2]] : !forth.stack -> !forth.stack, i64
\ CHECK-NEXT:  %[[OS2:.*]], %[[LIM:.*]] = forth.pop %[[OS]] : !forth.stack -> !forth.stack, i64
\ CHECK:       cf.br ^bb1(%[[OS2]] : !forth.stack)
\ CHECK:     ^bb1(%[[B1:.*]]: !forth.stack):
\ CHECK:       %[[STEP_S:.*]] = forth.constant %[[B1]](-1 : i32) : !forth.stack -> !forth.stack
\ CHECK-NEXT:  %[[POP_S:.*]], %[[STEP64:.*]] = forth.pop %[[STEP_S]] : !forth.stack -> !forth.stack, i64
\ CHECK-NEXT:  %[[STEP:.*]] = arith.trunci %[[STEP64]] : i64 to i32
\ CHECK:       %[[OLD:.*]] = memref.load
\ CHECK:       %[[NEW:.*]] = arith.addi %[[OLD]], %[[STEP]] : i32
\ CHECK:       %[[D1:.*]] = arith.subi %[[OLD]], %{{.*}} : i32
\ CHECK-NEXT:  %[[D2:.*]] = arith.subi %[[NEW]], %{{.*}} : i32
\ CHECK-NEXT:  %[[XOR:.*]] = arith.xori %[[D1]], %[[D2]] : i32
\ CHECK-NEXT:  %[[ZERO:.*]] = arith.constant 0 : i32
\ CHECK-NEXT:  %[[CROSSED:.*]] = arith.cmpi slt, %[[XOR]], %[[ZERO]] : i32
\ CHECK-NEXT:  cf.cond_br %[[CROSSED]], ^bb2(%[[POP_S]] : !forth.stack), ^bb1(%[[POP_S]] : !forth.stack)
\ CHECK:     ^bb2(%{{.*}}: !forth.stack):
\ CHECK-NEXT:  return
\! kernel main
0 10 DO -1 +LOOP
