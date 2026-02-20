\ RUN: %warpforth-translate --forth-to-mlir %s | %FileCheck %s

\ Test @ produces forth.loadi
\ CHECK: forth.loadi %{{.*}} : !forth.stack -> !forth.stack

\ Test ! produces forth.storei
\ CHECK: forth.storei %{{.*}} : !forth.stack -> !forth.stack

\ Test S@ produces forth.shared_loadi
\ CHECK: forth.shared_loadi %{{.*}} : !forth.stack -> !forth.stack

\ Test S! produces forth.shared_storei
\ CHECK: forth.shared_storei %{{.*}} : !forth.stack -> !forth.stack

\ Test CELLS produces literal 8 + mul
\ CHECK: forth.constant %{{.*}}(8 : i64)
\ CHECK-NEXT: forth.muli
\! kernel main
1 @ 2 3 ! 4 S@ 5 6 S!
4 CELLS
