\ RUN: %warpforth-translate --forth-to-mlir %s | %FileCheck %s

\ Verify bitwise operations parse correctly with SSA chaining
\ CHECK: %[[S0:.*]] = forth.stack
\ CHECK: %[[S1:.*]] = forth.literal %[[S0]]
\ CHECK: %[[S2:.*]] = forth.literal %[[S1]]
\ CHECK: %[[S3:.*]] = forth.and %[[S2]]
\ CHECK: %[[S4:.*]] = forth.literal %[[S3]]
\ CHECK: %[[S5:.*]] = forth.literal %[[S4]]
\ CHECK: %[[S6:.*]] = forth.or %[[S5]]
\ CHECK: %[[S7:.*]] = forth.literal %[[S6]]
\ CHECK: %[[S8:.*]] = forth.literal %[[S7]]
\ CHECK: %[[S9:.*]] = forth.xor %[[S8]]
\ CHECK: %[[S10:.*]] = forth.literal %[[S9]]
\ CHECK: %[[S11:.*]] = forth.not %[[S10]]
\ CHECK: %[[S12:.*]] = forth.literal %[[S11]]
\ CHECK: %[[S13:.*]] = forth.literal %[[S12]]
\ CHECK: %[[S14:.*]] = forth.lshift %[[S13]]
\ CHECK: %[[S15:.*]] = forth.literal %[[S14]]
\ CHECK: %[[S16:.*]] = forth.literal %[[S15]]
\ CHECK: %{{.*}} = forth.rshift %[[S16]]
3 5 AND 7 8 OR 15 3 XOR 42 NOT 1 4 LSHIFT 256 2 RSHIFT
