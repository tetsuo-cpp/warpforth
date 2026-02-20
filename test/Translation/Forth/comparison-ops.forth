\ RUN: %warpforth-translate --forth-to-mlir %s | %FileCheck %s

\ Verify comparison operations parse correctly
\ CHECK: %[[S0:.*]] = forth.stack
\ CHECK: %[[S1:.*]] = forth.constant %[[S0]]
\ CHECK: %[[S2:.*]] = forth.constant %[[S1]]
\ CHECK: %[[S3:.*]] = forth.eqi %[[S2]]
\ CHECK: %[[S4:.*]] = forth.constant %[[S3]]
\ CHECK: %[[S5:.*]] = forth.constant %[[S4]]
\ CHECK: %[[S6:.*]] = forth.lti %[[S5]]
\ CHECK: %[[S7:.*]] = forth.constant %[[S6]]
\ CHECK: %[[S8:.*]] = forth.constant %[[S7]]
\ CHECK: %[[S9:.*]] = forth.gti %[[S8]]
\ CHECK: %[[S10:.*]] = forth.constant %[[S9]]
\ CHECK: %[[S11:.*]] = forth.zero_eq %[[S10]]
\ CHECK: %[[S12:.*]] = forth.constant %[[S11]]
\ CHECK: %[[S13:.*]] = forth.constant %[[S12]]
\ CHECK: %[[S14:.*]] = forth.nei %[[S13]]
\ CHECK: %[[S15:.*]] = forth.constant %[[S14]]
\ CHECK: %[[S16:.*]] = forth.constant %[[S15]]
\ CHECK: %[[S17:.*]] = forth.lei %[[S16]]
\ CHECK: %[[S18:.*]] = forth.constant %[[S17]]
\ CHECK: %[[S19:.*]] = forth.constant %[[S18]]
\ CHECK: %{{.*}} = forth.gei %[[S19]]
\! kernel main
1 2 = 3 4 < 5 6 > 0 0= 7 8 <> 9 10 <= 11 12 >=
