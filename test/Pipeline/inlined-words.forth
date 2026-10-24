\ RUN: %warpforthc %s -o %t.ptx
\ RUN: %FileCheck %s < %t.ptx

\ Words inlined into the kernel must not survive as device functions.
\ CHECK-NOT: .func
\ CHECK: .entry k(
\ CHECK-NOT: .func

\! kernel k
\! param a i64[256]
: sq DUP * ;
: sumsq 0 SWAP 0 DO I sq + LOOP ;
GLOBAL-ID sumsq GLOBAL-ID CELLS a + !
