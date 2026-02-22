\ RUN: %warpforth-translate --forth-to-mlir %s | %FileCheck %s

\ Verify scalar param uses i32 argument type.
\ CHECK: func.func private @main(%arg0: i32 {forth.param_name = "SCALE"})
\ CHECK: forth.param_ref %{{.*}} "SCALE"
\! kernel main
\! param SCALE i32
SCALE
