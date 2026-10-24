\ RUN: %not %warpforth-translate --forth-to-mlir %s 2>&1 | %FileCheck %s
\ CHECK: {{.*}}hex-literal-error.forth:5:1: error: unknown word: 0X10
\! kernel k
\! param a i64[1]
0x10 a !
