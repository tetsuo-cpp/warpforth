\ RUN: %warpforth-translate --forth-to-mlir %s | %FileCheck %s

\ CHECK: forth.stack
\ CHECK-NEXT: forth.constant %{{.*}}(42 : i64)
\ CHECK-NEXT: forth.constant %{{.*}}(-7 : i64)
\ CHECK-NEXT: forth.constant %{{.*}}(0 : i64)
\! kernel main
42 -7 0
