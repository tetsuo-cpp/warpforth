\ RUN: %warpforth-translate --forth-to-mlir %s | %FileCheck %s

\ Verify conditional LEAVE preserves the loop backedge for non-LEAVE paths.

\ Branch directly to body (post-test loop)
\ CHECK:       cf.br ^bb1(%{{.*}} : !forth.stack)

\ Body: I 5 = IF → cond_br to LEAVE or THEN merge
\ CHECK:     ^bb1(%[[B:.*]]: !forth.stack):
\ CHECK:       forth.pop_flag
\ CHECK-NEXT:  cf.cond_br %{{[^,]*}}, ^bb[[LEAVE:[0-9]+]](%{{[^)]*}} : !forth.stack), ^bb[[JOIN:[0-9]+]](%{{[^)]*}} : !forth.stack)

\ Exit: return
\ CHECK:     ^bb[[EXIT:[0-9]+]](%{{.*}}: !forth.stack):
\ CHECK:       return

\ LEAVE branch: unconditional jump to exit
\ CHECK:     ^bb[[LEAVE]](%{{.*}}: !forth.stack):
\ CHECK:       cf.cond_br %true, ^bb[[EXIT]](%{{.*}} : !forth.stack), ^bb[[DEAD:[0-9]+]](%{{.*}} : !forth.stack)

\ Join (THEN merge): 1 DROP, crossing test, loop back to body or exit
\ CHECK:     ^bb[[JOIN]](%{{.*}}: !forth.stack):
\ CHECK:       arith.xori
\ CHECK:       arith.cmpi slt
\ CHECK:       cf.cond_br

\ Dead block from LEAVE
\ CHECK:     ^bb[[DEAD]](%{{.*}}: !forth.stack):
\ CHECK:       cf.br ^bb[[JOIN]]

\! kernel main
10 0 DO
  I 5 = IF LEAVE THEN
  1 DROP
LOOP
