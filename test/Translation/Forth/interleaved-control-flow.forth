\ RUN: %warpforth-translate --forth-to-mlir %s | %FileCheck %s

\ These test interleaved control flow patterns that require a compile-time
\ control-flow stack (cf-stack) and cannot be expressed with structured
\ region-holding ops.

\ === Multi-WHILE: two exit conditions from the same loop ===
\ Loop exits via WHILE(1) when value <= 10, or via WHILE(2) when value is odd.
\ The DROP between REPEAT and THEN cleans up the stack on the WHILE(2) exit path.

\ CHECK-LABEL: func.func private @MULTI_WHILE
\ Entry: branch to loop header
\ CHECK:       cf.br ^bb1

\ Loop header: DUP 10 > → WHILE(1)
\ CHECK:     ^bb1(%[[H:.*]]: !forth.stack):
\ CHECK:       forth.dup
\ CHECK:       forth.constant %{{.*}}(10 : i64)
\ CHECK-NEXT:  %{{.*}} = forth.gti
\ CHECK:       forth.pop_flag
\ CHECK-NEXT:  cf.cond_br %{{.*}}, ^bb2(%{{.*}} : !forth.stack), ^bb3(%{{.*}} : !forth.stack)

\ WHILE(1) body: DUP 2 MOD 0= → WHILE(2)
\ CHECK:     ^bb2(%{{.*}}: !forth.stack):
\ CHECK:       forth.dup
\ CHECK:       forth.constant %{{.*}}(2 : i64)
\ CHECK-NEXT:  %{{.*}} = forth.mod
\ CHECK-NEXT:  %{{.*}} = forth.zero_eq
\ CHECK:       forth.pop_flag
\ CHECK-NEXT:  cf.cond_br %{{.*}}, ^bb4(%{{.*}} : !forth.stack), ^bb5(%{{.*}} : !forth.stack)

\ WHILE(1) exit / return block (also reached from THEN)
\ CHECK:     ^bb3(%{{.*}}: !forth.stack):
\ CHECK-NEXT:  return

\ WHILE(2) body: 1 - → REPEAT (branch back to loop header)
\ CHECK:     ^bb4(%[[B4:.*]]: !forth.stack):
\ CHECK-NEXT:  %{{.*}} = forth.constant %[[B4]](1 : i64)
\ CHECK-NEXT:  %{{.*}} = forth.subi
\ CHECK-NEXT:  cf.br ^bb1

\ WHILE(2) exit: DROP → THEN (branch to WHILE(1) exit)
\ CHECK:     ^bb5(%{{.*}}: !forth.stack):
\ CHECK-NEXT:  %{{.*}} = forth.drop
\ CHECK-NEXT:  cf.br ^bb3

\! kernel main
: multi-while
  BEGIN DUP 10 > WHILE DUP 2 MOD 0= WHILE 1 - REPEAT DROP THEN ;

\ === WHILE+UNTIL: two different exit mechanisms from the same loop ===
\ WHILE checks the pre-condition (value > 0), UNTIL checks a post-condition
\ (value = 5). The loop has two distinct exit paths that merge at THEN.

\ CHECK-LABEL: func.func private @WHILE_UNTIL
\ Entry: branch to loop header
\ CHECK:       cf.br ^bb1

\ Loop header: DUP 0 > → WHILE
\ CHECK:     ^bb1(%{{.*}}: !forth.stack):
\ CHECK:       forth.dup
\ CHECK:       forth.constant %{{.*}}(0 : i64)
\ CHECK-NEXT:  %{{.*}} = forth.gti
\ CHECK:       forth.pop_flag
\ CHECK-NEXT:  cf.cond_br %{{.*}}, ^bb2(%{{.*}} : !forth.stack), ^bb3(%{{.*}} : !forth.stack)

\ WHILE body + UNTIL: 1 - DUP 5 = UNTIL
\ UNTIL true exits to ^bb4, UNTIL false loops back to ^bb1
\ CHECK:     ^bb2(%[[W:.*]]: !forth.stack):
\ CHECK-NEXT:  %{{.*}} = forth.constant %[[W]](1 : i64)
\ CHECK-NEXT:  %{{.*}} = forth.subi
\ CHECK:       forth.dup
\ CHECK:       forth.constant %{{.*}}(5 : i64)
\ CHECK-NEXT:  %{{.*}} = forth.eqi
\ CHECK:       forth.pop_flag
\ CHECK-NEXT:  cf.cond_br %{{.*}}, ^bb4(%{{.*}} : !forth.stack), ^bb1(%{{.*}} : !forth.stack)

\ WHILE exit / return block (also reached from THEN after UNTIL exit)
\ CHECK:     ^bb3(%{{.*}}: !forth.stack):
\ CHECK-NEXT:  return

\ UNTIL exit → THEN (branch to WHILE exit)
\ CHECK:     ^bb4(%{{.*}}: !forth.stack):
\ CHECK-NEXT:  cf.br ^bb3

: while-until
  BEGIN DUP 0 > WHILE 1 - DUP 5 = UNTIL THEN ;

multi-while while-until
