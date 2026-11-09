\ RUN: %warpforthc -O0 %s -o %t.ptx
\ RUN: %FileCheck %s < %t.ptx

\ An unused word that only calls another unused word is not part of the kernel.
\ CHECK-NOT: .func
\ CHECK: .entry main(
\ CHECK: st.global.u64
\ CHECK-NOT: .func

\! kernel main
\! param OUT i64[1]
: a 1 + ;
: b a ;
5 OUT !
