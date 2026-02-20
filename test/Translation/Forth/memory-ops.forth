\ RUN: %warpforth-translate --forth-to-mlir %s | %FileCheck %s

\ Test @ produces forth.load
\ CHECK: forth.load %{{.*}} : !forth.stack -> !forth.stack

\ Test ! produces forth.store
\ CHECK: forth.store %{{.*}} : !forth.stack -> !forth.stack

\ Test S@ produces forth.shared_load
\ CHECK: forth.shared_load %{{.*}} : !forth.stack -> !forth.stack

\ Test S! produces forth.shared_store
\ CHECK: forth.shared_store %{{.*}} : !forth.stack -> !forth.stack

\ Test CELLS produces literal 8 + mul
\ CHECK: forth.literal %{{.*}} 8
\ CHECK-NEXT: forth.mul
\! kernel main
1 @ 2 3 ! 4 S@ 5 6 S!
4 CELLS
