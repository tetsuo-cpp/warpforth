\ RUN: %not %warpforth-translate --forth-to-mlir %s 2>&1 | %FileCheck %s
\ CHECK: :[[@LINE+1]]:1: error: kernel name must be a valid identifier ([A-Za-z_][A-Za-z0-9_]*)
\! kernel my-kernel
\! param a i64[1]
1 a !
