\ RUN: %warpforth-translate --forth-to-mlir %s | %FileCheck %s

\ Verify SSA chaining: each op consumes the previous stack value
\ CHECK: %[[S0:.*]] = forth.stack
\ CHECK: %[[S1:.*]] = forth.literal %[[S0]]
\ CHECK: %[[S2:.*]] = forth.literal %[[S1]]
\ CHECK: %[[S3:.*]] = forth.add %[[S2]]
\ CHECK: %[[S4:.*]] = forth.sub %[[S3]]
\ CHECK: %[[S5:.*]] = forth.mul %[[S4]]
\ CHECK: %[[S6:.*]] = forth.div %[[S5]]
\ CHECK: %{{.*}} = forth.mod %[[S6]]
\! kernel main
1 2 + - * / MOD
