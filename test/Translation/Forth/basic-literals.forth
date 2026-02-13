\ RUN: %warpforth-translate --forth-to-mlir %s | %FileCheck %s

\ CHECK: forth.stack
\ CHECK-NEXT: forth.literal %{{.*}} 42
\ CHECK-NEXT: forth.literal %{{.*}} -7
\ CHECK-NEXT: forth.literal %{{.*}} 0
42 -7 0
