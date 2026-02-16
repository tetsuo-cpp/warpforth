\ RUN: %warpforth-translate --forth-to-mlir %s | %FileCheck %s

\ Verify comparison operations parse correctly
\ CHECK: %[[S0:.*]] = forth.stack
\ CHECK: %[[S1:.*]] = forth.literal %[[S0]]
\ CHECK: %[[S2:.*]] = forth.literal %[[S1]]
\ CHECK: %[[S3:.*]] = forth.eq %[[S2]]
\ CHECK: %[[S4:.*]] = forth.literal %[[S3]]
\ CHECK: %[[S5:.*]] = forth.literal %[[S4]]
\ CHECK: %[[S6:.*]] = forth.lt %[[S5]]
\ CHECK: %[[S7:.*]] = forth.literal %[[S6]]
\ CHECK: %[[S8:.*]] = forth.literal %[[S7]]
\ CHECK: %[[S9:.*]] = forth.gt %[[S8]]
\ CHECK: %[[S10:.*]] = forth.literal %[[S9]]
\ CHECK: %[[S11:.*]] = forth.zero_eq %[[S10]]
\ CHECK: %[[S12:.*]] = forth.literal %[[S11]]
\ CHECK: %[[S13:.*]] = forth.literal %[[S12]]
\ CHECK: %[[S14:.*]] = forth.ne %[[S13]]
\ CHECK: %[[S15:.*]] = forth.literal %[[S14]]
\ CHECK: %[[S16:.*]] = forth.literal %[[S15]]
\ CHECK: %[[S17:.*]] = forth.le %[[S16]]
\ CHECK: %[[S18:.*]] = forth.literal %[[S17]]
\ CHECK: %[[S19:.*]] = forth.literal %[[S18]]
\ CHECK: %{{.*}} = forth.ge %[[S19]]
1 2 = 3 4 < 5 6 > 0 0= 7 8 <> 9 10 <= 11 12 >=
