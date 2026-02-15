\ RUN: %warpforth-translate --forth-to-mlir %s | %FileCheck %s

\ Test @ produces forth.load
\ CHECK: forth.load %{{.*}} : !forth.stack -> !forth.stack

\ Test ! produces forth.store
\ CHECK: forth.store %{{.*}} : !forth.stack -> !forth.stack

\ Test CELLS produces literal 8 + mul
\ CHECK: forth.literal %{{.*}} 8
\ CHECK-NEXT: forth.mul
1 @ 2 3 !
4 CELLS
