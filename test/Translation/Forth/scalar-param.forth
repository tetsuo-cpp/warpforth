\ RUN: %warpforth-translate --forth-to-mlir %s | %FileCheck %s

\ Verify scalar param uses i64 argument type.
\ CHECK: func.func private @main(%arg0: i64 {forth.param_name = "SCALE"})
\ CHECK: forth.param_ref %{{.*}} %arg0 : !forth.stack, i64 -> !forth.stack
\! kernel main
\! param SCALE i64
SCALE
