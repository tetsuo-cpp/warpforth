\ RUN: %warpforth-translate --forth-to-mlir %s | %FileCheck %s

\ Verify float arithmetic ops parse correctly
\ CHECK: %[[S0:.*]] = forth.stack
\ CHECK: %[[S1:.*]] = forth.constant %[[S0]]
\ CHECK: %[[S2:.*]] = forth.constant %[[S1]]
\ CHECK: %[[S3:.*]] = forth.addf %[[S2]]
\ CHECK: %[[S4:.*]] = forth.subf %[[S3]]
\ CHECK: %[[S5:.*]] = forth.mulf %[[S4]]
\ CHECK: %{{.*}} = forth.divf %[[S5]]
\! kernel main
1.0 2.0 F+ F- F* F/
