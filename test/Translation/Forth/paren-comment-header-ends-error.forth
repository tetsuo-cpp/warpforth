\ RUN: %not %warpforth-translate --forth-to-mlir %s 2>&1 | %FileCheck %s

\ Code after a paren comment ends the header, like any other code line.
\ CHECK: [[@LINE+3]]:1: error: header directive must appear before any code
\! kernel main
( comment ) 1
\! param A i64[4]
