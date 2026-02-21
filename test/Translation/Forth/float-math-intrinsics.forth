\ RUN: %warpforth-translate --forth-to-mlir %s | %FileCheck %s

\ Verify float math intrinsic ops parse correctly

\ Unary ops
\ CHECK: %[[S0:.*]] = forth.stack
\ CHECK: %[[S1:.*]] = forth.constant %[[S0]]
\ CHECK: %[[S2:.*]] = forth.expf %[[S1]]
\ CHECK: %[[S3:.*]] = forth.sqrtf %[[S2]]
\ CHECK: %[[S4:.*]] = forth.logf %[[S3]]
\ CHECK: %[[S5:.*]] = forth.absf %[[S4]]
\ CHECK: %[[S6:.*]] = forth.negf %[[S5]]

\ Binary ops
\ CHECK: %[[S7:.*]] = forth.constant %[[S6]]
\ CHECK: %[[S8:.*]] = forth.maxf %[[S7]]
\ CHECK: %[[S9:.*]] = forth.minf %[[S8]]

\! kernel main
1.0 FEXP FSQRT FLOG FABS FNEG
2.0 FMAX FMIN
