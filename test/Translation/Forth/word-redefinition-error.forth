\ RUN: %not %warpforth-translate --forth-to-mlir %s 2>&1 | %FileCheck %s
\ CHECK: :[[@LINE+6]]:3: error: word 'FOO' is already defined
\ CHECK: :[[@LINE+4]]:3: note: previous definition is here
\ CHECK-NOT: redefinition of symbol
\! kernel k
\! param a i64[1]
: foo 1 ;
: foo 2 ;
foo a !
