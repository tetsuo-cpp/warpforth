\ RUN: %warpforth-translate --forth-to-mlir %s | %FileCheck %s

\ Test F@ produces forth.loadf
\ CHECK: forth.loadf %{{.*}} : !forth.stack -> !forth.stack

\ Test F! produces forth.storef
\ CHECK: forth.storef %{{.*}} : !forth.stack -> !forth.stack

\ Test SF@ produces forth.shared_loadf
\ CHECK: forth.shared_loadf %{{.*}} : !forth.stack -> !forth.stack

\ Test SF! produces forth.shared_storef
\ CHECK: forth.shared_storef %{{.*}} : !forth.stack -> !forth.stack
\! kernel main
1 F@ 2.0 3 F! 4 SF@ 5.0 6 SF!
