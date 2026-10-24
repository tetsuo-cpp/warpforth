\ RUN: %warpforth-translate --forth-to-mlir %s | %FileCheck %s

\ CHECK: forth.stack
\ CHECK-NEXT: forth.constant %{{.*}}(3.140000e+00 : f64)
\ CHECK-NEXT: forth.constant %{{.*}}(-2.000000e+00 : f64)
\ CHECK-NEXT: forth.constant %{{.*}}(1.000000e-05 : f64)
\ CHECK-NEXT: forth.constant %{{.*}}(1.000000e+03 : f64)
\ CHECK-NEXT: forth.constant %{{.*}}(1.000000e+03 : f64)
\ CHECK-NEXT: forth.constant %{{.*}}(5.000000e-01 : f64)
\ CHECK-NEXT: forth.constant %{{.*}}(-5.000000e-01 : f64)
\ CHECK-NEXT: forth.constant %{{.*}}(5.000000e+00 : f64)
\ CHECK-NEXT: forth.constant %{{.*}}(1.500000e+00 : f64)
\ CHECK-NEXT: forth.constant %{{.*}}(2.500000e+02 : f64)
\ CHECK-NEXT: forth.constant %{{.*}}(1.7976931348623157E+308 : f64)
\ CHECK-NEXT: forth.constant %{{.*}}(0.000000e+00 : f64)
\! kernel main
3.14 -2.0 1.0e-5 1e3 1E3 .5 -.5 5. +1.5 2.5e+2 1.7976931348623157e308 1e-400
