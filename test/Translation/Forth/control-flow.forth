\ RUN: %warpforth-translate --forth-to-mlir %s | %FileCheck %s

\ Verify IF/ELSE/THEN parsing produces forth.if with block-arg regions

\ Basic IF/ELSE/THEN
\ CHECK: %[[S0:.*]] = forth.stack
\ CHECK: %[[S1:.*]] = forth.literal %[[S0]] 1
\ CHECK: %[[IF1:.*]] = forth.if %[[S1]]
\ CHECK:   ^bb0(%[[ARG1:.*]]: !forth.stack):
\ CHECK:   forth.drop %[[ARG1]]
\ CHECK:   forth.literal %{{.*}} 42
\ CHECK:   forth.yield
\ CHECK: } else {
\ CHECK:   ^bb0(%[[ARG2:.*]]: !forth.stack):
\ CHECK:   forth.drop %[[ARG2]]
\ CHECK:   forth.literal %{{.*}} 99
\ CHECK:   forth.yield
\ CHECK: }
1 IF 42 ELSE 99 THEN

\ Basic IF/THEN (no ELSE — identity drop+yield in else region)
\ CHECK: %[[S2:.*]] = forth.literal %[[IF1]] 0
\ CHECK: %[[IF2:.*]] = forth.if %[[S2]]
\ CHECK:   ^bb0(%[[ARG3:.*]]: !forth.stack):
\ CHECK:   forth.drop %[[ARG3]]
\ CHECK:   forth.literal %{{.*}} 7
\ CHECK:   forth.yield
\ CHECK: } else {
\ CHECK:   ^bb0(%[[ARG4:.*]]: !forth.stack):
\ CHECK:   forth.drop %[[ARG4]]
\ CHECK:   forth.yield
\ CHECK: }
0 IF 7 THEN
