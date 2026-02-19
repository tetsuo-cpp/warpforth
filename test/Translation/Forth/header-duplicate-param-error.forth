\ RUN: %not %warpforth-translate --forth-to-mlir %s 2>&1 | %FileCheck %s
\ CHECK: duplicate parameter name: A
\! kernel main
\! param A i64[4]
\! param A i64[8]
