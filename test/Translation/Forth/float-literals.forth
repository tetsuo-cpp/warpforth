\ RUN: %warpforth-translate --forth-to-mlir %s | %FileCheck %s

\ CHECK: forth.stack
\ CHECK-NEXT: forth.constant %{{.*}}(3.140000e+00 : f64)
\ CHECK-NEXT: forth.constant %{{.*}}(-2.000000e+00 : f64)
\ CHECK-NEXT: forth.constant %{{.*}}(1.000000e-05 : f64)
\ CHECK-NEXT: forth.constant %{{.*}}(1.000000e+03 : f64)
\! kernel main
3.14 -2.0 1.0e-5 1e3
