\ RUN: %warpforth-translate --forth-to-mlir %s | %FileCheck %s

\ Check f64 scalar param becomes f64 function argument
\ CHECK: func.func private @main(%arg0: memref<256xf64> {forth.param_name = "DATA"}, %arg1: f64 {forth.param_name = "SCALE"})

\ Check param refs work
\ CHECK: forth.param_ref %{{.*}} "DATA"
\ CHECK: forth.param_ref %{{.*}} "SCALE"
\! kernel main
\! param DATA f64[256]
\! param SCALE f64
DATA SCALE
