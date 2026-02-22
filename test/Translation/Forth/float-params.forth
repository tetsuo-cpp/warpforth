\ RUN: %warpforth-translate --forth-to-mlir %s | %FileCheck %s

\ Check f32 scalar param becomes f32 function argument
\ CHECK: func.func private @main(%arg0: memref<256xf32> {forth.param_name = "DATA"}, %arg1: f32 {forth.param_name = "SCALE"})

\ Check param refs work
\ CHECK: forth.param_ref %{{.*}} "DATA"
\ CHECK: forth.param_ref %{{.*}} "SCALE"
\! kernel main
\! param DATA f32[256]
\! param SCALE f32
DATA SCALE
