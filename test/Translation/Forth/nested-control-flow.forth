\ RUN: %warpforth-translate --forth-to-mlir %s | %FileCheck %s

\ === Nested IF ===
\ CHECK: %[[S0:.*]] = forth.stack
\ CHECK: %[[S1:.*]] = forth.literal %[[S0]] 1
\ CHECK: %[[IF1:.*]] = forth.if %[[S1]]
\ CHECK:   ^bb0(%[[A1:.*]]: !forth.stack):
\ CHECK:   %[[D1:.*]] = forth.drop %[[A1]]
\ CHECK:   %[[L2:.*]] = forth.literal %[[D1]] 2
\ CHECK:   %[[IF2:.*]] = forth.if %[[L2]]
\ CHECK:     ^bb0(%[[A2:.*]]: !forth.stack):
\ CHECK:     forth.drop %[[A2]]
\ CHECK:     forth.literal %{{.*}} 3
\ CHECK:     forth.yield
\ CHECK:   } else {
\ CHECK:     ^bb0(%[[A3:.*]]: !forth.stack):
\ CHECK:     forth.drop %[[A3]]
\ CHECK:     forth.yield
\ CHECK:   }
\ CHECK:   forth.yield
\ CHECK: } else {
\ CHECK:   ^bb0(%[[A4:.*]]: !forth.stack):
\ CHECK:   forth.drop %[[A4]]
\ CHECK:   forth.yield
\ CHECK: }
1 IF 2 IF 3 THEN THEN

\ === IF inside DO ===
\ CHECK: %[[S2:.*]] = forth.literal %[[IF1]] 10
\ CHECK: %[[S3:.*]] = forth.literal %[[S2]] 0
\ CHECK: %[[LOOP1:.*]] = forth.do_loop %[[S3]]
\ CHECK:   ^bb0(%[[BA:.*]]: !forth.stack):
\ CHECK:   %[[LI:.*]] = forth.loop_index %[[BA]]
\ CHECK:   %[[L5:.*]] = forth.literal %[[LI]] 5
\ CHECK:   %[[GT:.*]] = forth.gt %[[L5]]
\ CHECK:   forth.if %[[GT]]
\ CHECK:     forth.loop_index
\ CHECK:   forth.yield
10 0 DO I 5 > IF I THEN LOOP

\ === Nested DO with J ===
\ CHECK: forth.do_loop
\ CHECK:   forth.do_loop
\ CHECK:     forth.loop_index %{{.*}} {depth = 1 : i64}
\ CHECK:     forth.loop_index %{{.*}}
\ CHECK:     forth.add
3 0 DO 4 0 DO J I + LOOP LOOP

\ === Triple-nested DO with K ===
\ CHECK: forth.do_loop
\ CHECK:   forth.do_loop
\ CHECK:     forth.do_loop
\ CHECK:       forth.loop_index %{{.*}} {depth = 2 : i64}
\ CHECK:       forth.loop_index %{{.*}} {depth = 1 : i64}
\ CHECK:       forth.loop_index %{{.*}}
\ CHECK:       forth.add
\ CHECK:       forth.add
2 0 DO 2 0 DO 2 0 DO K J I + + LOOP LOOP LOOP

\ === BEGIN/WHILE inside IF ===
\ CHECK: forth.if
\ CHECK:   forth.begin_while_repeat
\ CHECK:     forth.dup
\ CHECK:     forth.yield %{{.*}} while_cond
\ CHECK:   } do {
\ CHECK:     forth.literal %{{.*}} 1
\ CHECK:     forth.sub
\ CHECK:     forth.yield
5 IF BEGIN DUP WHILE 1 - REPEAT THEN

\ === IF inside BEGIN/UNTIL ===
\ CHECK: forth.begin_until
\ CHECK:   forth.dup
\ CHECK:   forth.literal %{{.*}} 10
\ CHECK:   forth.lt
\ CHECK:   forth.if
\ CHECK:     forth.literal %{{.*}} 1
\ CHECK:     forth.add
\ CHECK:   forth.dup
\ CHECK:   forth.literal %{{.*}} 20
\ CHECK:   forth.eq
\ CHECK:   forth.yield
BEGIN DUP 10 < IF 1 + THEN DUP 20 = UNTIL
