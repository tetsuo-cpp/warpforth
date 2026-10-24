\ RUN: %not %warpforth-translate --forth-to-mlir %s 2>&1 | %FileCheck %s
\ CHECK: {{.*}}float-literal-overflow-error.forth:4:5: error: float literal out of range for f64
\! kernel main
1.0 -1e999
