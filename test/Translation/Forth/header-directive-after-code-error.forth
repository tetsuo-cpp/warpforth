\ RUN: %not %warpforth-translate --forth-to-mlir %s 2>&1 | %FileCheck %s
\ CHECK: header directive must appear before any code
\! kernel main
\! param A i64[4]
A @
\! param B i64[4]
