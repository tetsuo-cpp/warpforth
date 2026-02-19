\ RUN: %warpforth-translate --forth-to-mlir %s | %FileCheck %s

\ === Nested IF ===
\ CHECK:       %[[S0:.*]] = forth.stack !forth.stack
\ CHECK-NEXT:  %[[S1:.*]] = forth.literal %[[S0]] 1 : !forth.stack -> !forth.stack
\ CHECK-NEXT:  %[[PF1:.*]], %[[FL1:.*]] = forth.pop_flag %[[S1]] : !forth.stack -> !forth.stack, i1
\ CHECK-NEXT:  cf.cond_br %[[FL1]], ^bb1(%[[PF1]] : !forth.stack), ^bb2(%[[PF1]] : !forth.stack)
\ CHECK:     ^bb1(%[[B1:.*]]: !forth.stack):
\ CHECK-NEXT:  %[[L2:.*]] = forth.literal %[[B1]] 2 : !forth.stack -> !forth.stack
\ CHECK-NEXT:  %[[PF2:.*]], %[[FL2:.*]] = forth.pop_flag %[[L2]] : !forth.stack -> !forth.stack, i1
\ CHECK-NEXT:  cf.cond_br %[[FL2]], ^bb3(%[[PF2]] : !forth.stack), ^bb4(%[[PF2]] : !forth.stack)
1 IF 2 IF 3 THEN THEN

\ === IF inside DO ===
\ CHECK:     ^bb2(%[[B2:.*]]: !forth.stack):
\ CHECK-NEXT:  %[[L10:.*]] = forth.literal %[[B2]] 10 : !forth.stack -> !forth.stack
\ CHECK-NEXT:  %[[L0A:.*]] = forth.literal %[[L10]] 0 : !forth.stack -> !forth.stack
\ CHECK-NEXT:  %[[POP1:.*]], %[[V1:.*]] = forth.pop %[[L0A]] : !forth.stack -> !forth.stack, i64
\ CHECK-NEXT:  %[[POP2:.*]], %[[V2:.*]] = forth.pop %[[POP1]] : !forth.stack -> !forth.stack, i64
\ CHECK-NEXT:  %[[ALLOC1:.*]] = memref.alloca() : memref<1xi64>
\ CHECK-NEXT:  %{{.*}} = arith.constant 0 : index
\ CHECK-NEXT:  memref.store %[[V1]], %[[ALLOC1]][%{{.*}}] : memref<1xi64>
\ CHECK-NEXT:  cf.br ^bb5(%[[POP2]] : !forth.stack)
10 0 DO I 5 > IF I THEN LOOP

\ Nested IF: true branch pushes 3, then merges
\ CHECK:     ^bb3(%[[B3:.*]]: !forth.stack):
\ CHECK-NEXT:  %[[L3:.*]] = forth.literal %[[B3]] 3 : !forth.stack -> !forth.stack
\ CHECK-NEXT:  cf.br ^bb4(%[[L3]] : !forth.stack)
\ CHECK:     ^bb4(%[[B4:.*]]: !forth.stack):
\ CHECK-NEXT:  cf.br ^bb2(%[[B4]] : !forth.stack)

\ DO loop header: check index < limit
\ CHECK:     ^bb5(%[[B5:.*]]: !forth.stack):
\ CHECK:       arith.cmpi slt
\ CHECK-NEXT:  cf.cond_br %{{.*}}, ^bb6(%[[B5]] : !forth.stack), ^bb7(%[[B5]] : !forth.stack)

\ DO loop body: I 5 > IF I THEN
\ CHECK:     ^bb6(%[[B6:.*]]: !forth.stack):
\ CHECK:       forth.push_value %[[B6]]
\ CHECK:       forth.literal %{{.*}} 5
\ CHECK-NEXT:  %{{.*}} = forth.gt
\ CHECK:       forth.pop_flag
\ CHECK-NEXT:  cf.cond_br %{{.*}}, ^bb8(%{{.*}} : !forth.stack), ^bb9(%{{.*}} : !forth.stack)

\ === Nested DO with J ===
\ After first DO loop exits: bb7 sets up nested DO (3 0 DO)
\ CHECK:     ^bb7(%[[B7:.*]]: !forth.stack):
\ CHECK-NEXT:  %{{.*}} = forth.literal %[[B7]] 3
3 0 DO 4 0 DO J I + LOOP LOOP

\ IF I true branch: push loop index
\ CHECK:     ^bb8(%[[B8:.*]]: !forth.stack):
\ CHECK:       forth.push_value %[[B8]]
\ CHECK-NEXT:  cf.br ^bb9

\ Loop increment and back-edge
\ CHECK:     ^bb9(%{{.*}}: !forth.stack):
\ CHECK:       arith.addi
\ CHECK:       memref.store
\ CHECK:       cf.br ^bb5

\ Outer DO loop (3 0 DO) header
\ CHECK:     ^bb10(%{{.*}}: !forth.stack):
\ CHECK:       arith.cmpi slt
\ CHECK:       cf.cond_br

\ Inner DO setup (4 0 DO)
\ CHECK:     ^bb11(%{{.*}}: !forth.stack):
\ CHECK:       forth.literal %{{.*}} 4
\ CHECK:       forth.literal %{{.*}} 0
\ CHECK:       forth.pop
\ CHECK:       forth.pop
\ CHECK:       memref.alloca()

\ === Triple-nested DO with K ===
\ After nested DO exits: sets up triple-nested DO (2 0 DO)
\ CHECK:     ^bb12(%{{.*}}: !forth.stack):
\ CHECK:       forth.literal %{{.*}} 2
2 0 DO 2 0 DO 2 0 DO K J I + + LOOP LOOP LOOP

\ Inner loop of J I + (bb13 header, bb14 body)
\ CHECK:     ^bb13(%{{.*}}: !forth.stack):
\ CHECK:       arith.cmpi slt
\ CHECK:       cf.cond_br

\ J I + body
\ CHECK:     ^bb14(%{{.*}}: !forth.stack):
\ CHECK:       forth.push_value
\ CHECK:       forth.push_value
\ CHECK:       forth.add

\ Outer loop increment (bb15)
\ CHECK:     ^bb15(%{{.*}}: !forth.stack):
\ CHECK:       arith.addi
\ CHECK:       cf.br ^bb10

\ Triple-nested outer loop header (bb16)
\ CHECK:     ^bb16(%{{.*}}: !forth.stack):
\ CHECK:       arith.cmpi slt
\ CHECK:       cf.cond_br

\ Triple-nested middle loop setup (bb17)
\ CHECK:     ^bb17(%{{.*}}: !forth.stack):
\ CHECK:       forth.literal %{{.*}} 2
\ CHECK:       forth.literal %{{.*}} 0

\ === BEGIN/WHILE inside IF ===
\ After triple-nested exits: 5 IF BEGIN DUP WHILE 1 - REPEAT THEN
\ CHECK:     ^bb18(%{{.*}}: !forth.stack):
\ CHECK:       forth.literal %{{.*}} 5
\ CHECK:       forth.pop_flag
\ CHECK-NEXT:  cf.cond_br
5 IF BEGIN DUP WHILE 1 - REPEAT THEN

\ bb25: IF true branch -> jump to begin/while header
\ CHECK:     ^bb25(%{{.*}}: !forth.stack):
\ CHECK-NEXT:  cf.br ^bb27

\ bb26: IF false branch (and WHILE exit) -> jump to BEGIN/UNTIL
\ CHECK:     ^bb26(%{{.*}}: !forth.stack):
\ CHECK-NEXT:  cf.br ^bb30

\ WHILE condition: DUP + pop_flag
\ CHECK:     ^bb27(%{{.*}}: !forth.stack):
\ CHECK:       forth.dup
\ CHECK:       forth.pop_flag
\ CHECK-NEXT:  cf.cond_br

\ WHILE body: 1 -
\ CHECK:     ^bb28(%[[B28:.*]]: !forth.stack):
\ CHECK-NEXT:  %{{.*}} = forth.literal %[[B28]] 1
\ CHECK-NEXT:  %{{.*}} = forth.sub

\ === IF inside BEGIN/UNTIL ===
\ BEGIN/UNTIL header: DUP 10 <
\ CHECK:     ^bb30(%{{.*}}: !forth.stack):
\ CHECK:       forth.dup
\ CHECK:       forth.literal %{{.*}} 10
\ CHECK-NEXT:  %{{.*}} = forth.lt
BEGIN DUP 10 < IF 1 + THEN DUP 20 = UNTIL

\ IF true branch: 1 +
\ CHECK:     ^bb31(%[[B31:.*]]: !forth.stack):
\ CHECK-NEXT:  %{{.*}} = forth.literal %[[B31]] 1
\ CHECK-NEXT:  %{{.*}} = forth.add

\ UNTIL condition: DUP 20 =
\ CHECK:     ^bb32(%{{.*}}: !forth.stack):
\ CHECK:       forth.dup
\ CHECK:       forth.literal %{{.*}} 20
\ CHECK-NEXT:  %{{.*}} = forth.eq
\ CHECK:       forth.pop_flag
\ CHECK-NEXT:  cf.cond_br
