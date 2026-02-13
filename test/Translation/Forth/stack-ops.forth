\ RUN: %warpforth-translate --forth-to-mlir %s | %FileCheck %s

\ CHECK: forth.stack
\ CHECK: forth.dup %{{.*}} : !forth.stack -> !forth.stack
\ CHECK: forth.drop %{{.*}} : !forth.stack -> !forth.stack
\ CHECK: forth.swap %{{.*}} : !forth.stack -> !forth.stack
\ CHECK: forth.over %{{.*}} : !forth.stack -> !forth.stack
\ CHECK: forth.rot %{{.*}} : !forth.stack -> !forth.stack
1 dup drop swap over rot
