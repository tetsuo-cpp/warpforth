\ RUN: %warpforth-translate --forth-to-mlir %s | %FileCheck %s

\ CHECK: forth.stack
\ CHECK-NEXT: forth.constant %{{.*}}(3.{{.*}} : f32)
\ CHECK-NEXT: forth.constant %{{.*}}(-2.{{.*}} : f32)
\ CHECK-NEXT: forth.constant %{{.*}}(9.{{.*}} : f32)
\ CHECK-NEXT: forth.constant %{{.*}}(1.{{.*}} : f32)
\! kernel main
3.14 -2.0 1.0e-5 1e3
