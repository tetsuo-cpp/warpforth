\ RUN: %not %warpforth-translate --forth-to-mlir %s 2>&1 | %FileCheck %s
\ CHECK: error: kernel name 'MY_K' conflicts with word MY-K
\! kernel MY_K
\! param a i64[1]
: my-k 1 ;
my-k a !
