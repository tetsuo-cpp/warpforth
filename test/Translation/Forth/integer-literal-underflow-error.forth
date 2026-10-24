\ RUN: %not %warpforth-translate --forth-to-mlir %s 2>&1 | %FileCheck %s
\ CHECK: {{.*}}integer-literal-underflow-error.forth:4:3: error: integer literal out of range for i64
\! kernel main
1 -9223372036854775809
