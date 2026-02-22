\ RUN: %warpforth-translate --forth-to-mlir %s | %FileCheck %s

\ === Nested IF ===
\ CHECK:       %[[S0:.*]] = forth.stack !forth.stack
\ CHECK-NEXT:  %[[S1:.*]] = forth.constant %[[S0]](1 : i32) : !forth.stack -> !forth.stack
\ CHECK-NEXT:  %[[PF1:.*]], %[[FL1:.*]] = forth.pop_flag %[[S1]] : !forth.stack -> !forth.stack, i1
\ CHECK-NEXT:  cf.cond_br %[[FL1]], ^bb1(%[[PF1]] : !forth.stack), ^bb2(%[[PF1]] : !forth.stack)
\ CHECK:     ^bb1(%[[B1:.*]]: !forth.stack):
\ CHECK-NEXT:  %[[L2:.*]] = forth.constant %[[B1]](2 : i32) : !forth.stack -> !forth.stack
\ CHECK-NEXT:  %[[PF2:.*]], %[[FL2:.*]] = forth.pop_flag %[[L2]] : !forth.stack -> !forth.stack, i1
\ CHECK-NEXT:  cf.cond_br %[[FL2]], ^bb3(%[[PF2]] : !forth.stack), ^bb4(%[[PF2]] : !forth.stack)
\! kernel main
1 IF 2 IF 3 THEN THEN

\ === IF inside DO ===
\ After IF/THEN merge, set up DO loop: 10 0 DO
\ CHECK:     ^bb2(%[[B2:.*]]: !forth.stack):
\ CHECK-NEXT:  %[[L10:.*]] = forth.constant %[[B2]](10 : i32) : !forth.stack -> !forth.stack
\ CHECK-NEXT:  %[[L0A:.*]] = forth.constant %[[L10]](0 : i32) : !forth.stack -> !forth.stack
\ CHECK-NEXT:  %[[POP1:.*]], %[[V1:.*]] = forth.pop %[[L0A]] : !forth.stack -> !forth.stack, i64
\ CHECK-NEXT:  %[[POP2:.*]], %[[V2:.*]] = forth.pop %[[POP1]] : !forth.stack -> !forth.stack, i64
\ CHECK-NEXT:  %[[TV1:.*]] = arith.trunci %[[V1]] : i64 to i32
\ CHECK-NEXT:  %[[TV2:.*]] = arith.trunci %[[V2]] : i64 to i32
\ CHECK-NEXT:  %[[ALLOC1:.*]] = memref.alloca() : memref<1xi32>
\ CHECK-NEXT:  %{{.*}} = arith.constant 0 : index
\ CHECK-NEXT:  memref.store %[[TV1]], %[[ALLOC1]][%{{.*}}] : memref<1xi32>
\ CHECK-NEXT:  cf.br ^bb5(%[[POP2]] : !forth.stack)
10 0 DO I 5 > IF I THEN LOOP

\ Nested IF: true branch pushes 3, then merges
\ CHECK:     ^bb3(%[[B3:.*]]: !forth.stack):
\ CHECK-NEXT:  %[[L3:.*]] = forth.constant %[[B3]](3 : i32) : !forth.stack -> !forth.stack
\ CHECK-NEXT:  cf.br ^bb4(%[[L3]] : !forth.stack)
\ CHECK:     ^bb4(%[[B4:.*]]: !forth.stack):
\ CHECK-NEXT:  cf.br ^bb2(%[[B4]] : !forth.stack)

\ DO loop body (post-test: no check block): I 5 > IF I THEN
\ CHECK:     ^bb5(%[[B5:.*]]: !forth.stack):
\ CHECK:       forth.push_value %[[B5]]
\ CHECK:       forth.constant %{{.*}}(5 : i32)
\ CHECK-NEXT:  %{{.*}} = forth.gti
\ CHECK:       forth.pop_flag
\ CHECK-NEXT:  cf.cond_br %{{[^,]*}}, ^bb7(%{{[^)]*}} : !forth.stack), ^bb8(%{{[^)]*}} : !forth.stack)

\ === Nested DO with J ===
\ After first DO loop exits: sets up nested DO (3 0 DO)
\ CHECK:     ^bb6(%[[B6:.*]]: !forth.stack):
\ CHECK-NEXT:  %{{.*}} = forth.constant %[[B6]](3 : i32)
3 0 DO 4 0 DO J I + LOOP LOOP

\ IF I true branch: push loop index
\ CHECK:     ^bb7(%[[B7:.*]]: !forth.stack):
\ CHECK:       forth.push_value %[[B7]]
\ CHECK-NEXT:  cf.br ^bb8

\ Loop end with crossing test and back-edge
\ CHECK:     ^bb8(%{{.*}}: !forth.stack):
\ CHECK:       arith.addi
\ CHECK:       memref.store
\ CHECK:       arith.xori
\ CHECK:       arith.cmpi slt
\ CHECK:       cf.cond_br

\ Outer DO body (3 0 DO) with inner DO setup (4 0 DO)
\ CHECK:     ^bb9(%{{.*}}: !forth.stack):
\ CHECK:       forth.constant %{{.*}}(4 : i32)
\ CHECK:       forth.constant %{{.*}}(0 : i32)
\ CHECK:       forth.pop
\ CHECK:       forth.pop
\ CHECK:       memref.alloca()

\ === Triple-nested DO with K ===
\ After nested DO exits: sets up triple-nested DO (2 0 DO)
\ CHECK:     ^bb10(%{{.*}}: !forth.stack):
\ CHECK:       forth.constant %{{.*}}(2 : i32)
2 0 DO 2 0 DO 2 0 DO K J I + + LOOP LOOP LOOP

\ Inner loop of J I + (bb11 body)
\ CHECK:     ^bb11(%{{.*}}: !forth.stack):
\ CHECK:       forth.push_value
\ CHECK:       forth.push_value
\ CHECK:       forth.addi

\ Inner loop crossing test
\ CHECK:       arith.xori
\ CHECK:       arith.cmpi slt
\ CHECK:       cf.cond_br

\ Outer loop increment (bb12)
\ CHECK:     ^bb12(%{{.*}}: !forth.stack):
\ CHECK:       arith.addi
\ CHECK:       arith.xori
\ CHECK:       arith.cmpi slt
\ CHECK:       cf.cond_br

\ Triple-nested outer loop body (bb13)
\ CHECK:     ^bb13(%{{.*}}: !forth.stack):
\ CHECK:       forth.constant %{{.*}}(2 : i32)
\ CHECK:       forth.constant %{{.*}}(0 : i32)

\ === BEGIN/WHILE inside IF ===
\ After triple-nested exits: 5 IF BEGIN DUP WHILE 1 - REPEAT THEN
\ CHECK:     ^bb14(%{{.*}}: !forth.stack):
\ CHECK:       forth.constant %{{.*}}(5 : i32)
\ CHECK:       forth.pop_flag
\ CHECK-NEXT:  cf.cond_br
5 IF BEGIN DUP WHILE 1 - REPEAT THEN

\ bb19: IF true branch -> jump to begin/while header
\ CHECK:     ^bb19(%{{.*}}: !forth.stack):
\ CHECK-NEXT:  cf.br ^bb21

\ bb20: IF false branch (and WHILE exit) -> jump to BEGIN/UNTIL
\ CHECK:     ^bb20(%{{.*}}: !forth.stack):
\ CHECK-NEXT:  cf.br ^bb24

\ WHILE condition: DUP + pop_flag
\ CHECK:     ^bb21(%{{.*}}: !forth.stack):
\ CHECK:       forth.dup
\ CHECK:       forth.pop_flag
\ CHECK-NEXT:  cf.cond_br

\ WHILE body: 1 -
\ CHECK:     ^bb22(%[[B22:.*]]: !forth.stack):
\ CHECK-NEXT:  %{{.*}} = forth.constant %[[B22]](1 : i32)
\ CHECK-NEXT:  %{{.*}} = forth.subi

\ === IF inside BEGIN/UNTIL ===
\ BEGIN/UNTIL header: DUP 10 <
\ CHECK:     ^bb24(%{{.*}}: !forth.stack):
\ CHECK:       forth.dup
\ CHECK:       forth.constant %{{.*}}(10 : i32)
\ CHECK-NEXT:  %{{.*}} = forth.lti
BEGIN DUP 10 < IF 1 + THEN DUP 20 = UNTIL

\ IF true branch: 1 +
\ CHECK:     ^bb25(%[[B25:.*]]: !forth.stack):
\ CHECK-NEXT:  %{{.*}} = forth.constant %[[B25]](1 : i32)
\ CHECK-NEXT:  %{{.*}} = forth.addi

\ UNTIL condition: DUP 20 =
\ CHECK:     ^bb26(%{{.*}}: !forth.stack):
\ CHECK:       forth.dup
\ CHECK:       forth.constant %{{.*}}(20 : i32)
\ CHECK-NEXT:  %{{.*}} = forth.eqi
\ CHECK:       forth.pop_flag
\ CHECK-NEXT:  cf.cond_br
