\ RUN: %warpforth-translate --forth-to-mlir %s | %FileCheck %s

\ CHECK: forth.stack
\ CHECK: forth.dup %{{.*}} : !forth.stack -> !forth.stack
\ CHECK: forth.drop %{{.*}} : !forth.stack -> !forth.stack
\ CHECK: forth.swap %{{.*}} : !forth.stack -> !forth.stack
\ CHECK: forth.over %{{.*}} : !forth.stack -> !forth.stack
\ CHECK: forth.rot %{{.*}} : !forth.stack -> !forth.stack
\ CHECK: forth.nip %{{.*}} : !forth.stack -> !forth.stack
\ CHECK: forth.tuck %{{.*}} : !forth.stack -> !forth.stack
\ CHECK: forth.pick %{{.*}} : !forth.stack -> !forth.stack
\ CHECK: forth.roll %{{.*}} : !forth.stack -> !forth.stack
\! kernel main
1 DUP DROP SWAP OVER ROT NIP TUCK PICK ROLL
