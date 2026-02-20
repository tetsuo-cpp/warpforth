\ RUN: %warpforth-translate --forth-to-mlir %s | %FileCheck %s

\ Verify float comparison operations parse correctly
\ CHECK: %[[S0:.*]] = forth.stack
\ CHECK: %[[S1:.*]] = forth.constant %[[S0]]
\ CHECK: %[[S2:.*]] = forth.constant %[[S1]]
\ CHECK: %[[S3:.*]] = forth.eqf %[[S2]]
\ CHECK: %[[S4:.*]] = forth.constant %[[S3]]
\ CHECK: %[[S5:.*]] = forth.constant %[[S4]]
\ CHECK: %[[S6:.*]] = forth.ltf %[[S5]]
\ CHECK: %[[S7:.*]] = forth.constant %[[S6]]
\ CHECK: %[[S8:.*]] = forth.constant %[[S7]]
\ CHECK: %[[S9:.*]] = forth.gtf %[[S8]]
\ CHECK: %[[S10:.*]] = forth.constant %[[S9]]
\ CHECK: %[[S11:.*]] = forth.constant %[[S10]]
\ CHECK: %[[S12:.*]] = forth.nef %[[S11]]
\ CHECK: %[[S13:.*]] = forth.constant %[[S12]]
\ CHECK: %[[S14:.*]] = forth.constant %[[S13]]
\ CHECK: %[[S15:.*]] = forth.lef %[[S14]]
\ CHECK: %[[S16:.*]] = forth.constant %[[S15]]
\ CHECK: %[[S17:.*]] = forth.constant %[[S16]]
\ CHECK: %{{.*}} = forth.gef %[[S17]]
\! kernel main
1.0 2.0 F= 3.0 4.0 F< 5.0 6.0 F> 7.0 8.0 F<> 9.0 10.0 F<= 11.0 12.0 F>=
