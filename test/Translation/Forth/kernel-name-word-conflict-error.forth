\ RUN: %not %warpforth-translate --forth-to-mlir %s 2>&1 | %FileCheck %s
\ CHECK: :[[@LINE+4]]:3: error: kernel name 'FOO' conflicts with word FOO
\ CHECK-NOT: redefinition of symbol
\! kernel FOO
\! param a i64[1]
: foo 1 ;
foo a !
