\ RUN: %warpforth-translate --forth-to-mlir %s | %FileCheck %s

\ Test S>F produces forth.itof
\ CHECK: forth.itof %{{.*}} : !forth.stack -> !forth.stack

\ Test F>S produces forth.ftoi
\ CHECK: forth.ftoi %{{.*}} : !forth.stack -> !forth.stack
\! kernel main
42 S>F F>S
