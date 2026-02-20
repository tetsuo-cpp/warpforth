\ RUN: %warpforth-translate --forth-to-mlir %s | %FileCheck %s

\ Verify SSA chaining: each op consumes the previous stack value
\ CHECK: %[[S0:.*]] = forth.stack
\ CHECK: %[[S1:.*]] = forth.constant %[[S0]]
\ CHECK: %[[S2:.*]] = forth.constant %[[S1]]
\ CHECK: %[[S3:.*]] = forth.addi %[[S2]]
\ CHECK: %[[S4:.*]] = forth.subi %[[S3]]
\ CHECK: %[[S5:.*]] = forth.muli %[[S4]]
\ CHECK: %[[S6:.*]] = forth.divi %[[S5]]
\ CHECK: %{{.*}} = forth.mod %[[S6]]
\! kernel main
1 2 + - * / MOD
