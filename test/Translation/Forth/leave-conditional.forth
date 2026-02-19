\ RUN: %warpforth-translate --forth-to-mlir %s | %FileCheck %s

\ Verify conditional LEAVE preserves the loop backedge for non-LEAVE paths.

\ CHECK:       cf.br ^bb1(%{{.*}} : !forth.stack)
\ CHECK:     ^bb1(%[[CHK:.*]]: !forth.stack):
\ CHECK:       cf.cond_br %{{.*}}, ^bb2(%[[CHK]] : !forth.stack), ^bb[[EXIT:[0-9]+]](%[[CHK]] : !forth.stack)
\ CHECK:     ^bb2(%[[B:.*]]: !forth.stack):
\ CHECK:       cf.cond_br %{{.*}}, ^bb[[LEAVE:[0-9]+]](%{{.*}} : !forth.stack), ^bb[[JOIN:[0-9]+]](%{{.*}} : !forth.stack)
\ CHECK:     ^bb[[EXIT]](%{{.*}}: !forth.stack):
\ CHECK:       return
\ CHECK:     ^bb[[LEAVE]](%{{.*}}: !forth.stack):
\ CHECK:       cf.cond_br %{{.*}}, ^bb[[EXIT]](%{{.*}} : !forth.stack), ^bb[[DEAD:[0-9]+]](%{{.*}} : !forth.stack)
\ CHECK:     ^bb[[JOIN]](%{{.*}}: !forth.stack):
\ CHECK:       cf.br ^bb1(%{{.*}} : !forth.stack)
\ CHECK:     ^bb[[DEAD]](%{{.*}}: !forth.stack):
\ CHECK:       cf.br ^bb[[JOIN]](%{{.*}} : !forth.stack)

10 0 DO
  I 5 = IF LEAVE THEN
  1 DROP
LOOP
