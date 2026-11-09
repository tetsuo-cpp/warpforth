\ RUN: %warpforthc %s -o %t.ptx
\ RUN: %FileCheck %s < %t.ptx

\ Words that read a GPU index or synchronize are inlined like any other word,
\ so they do not survive as device functions.
\ CHECK-NOT: .func
\ CHECK: .entry main(
\ CHECK: %tid.x
\ CHECK: bar.sync
\ CHECK: st.global
\ CHECK-NOT: .func

\! kernel main
\! param DATA i64[256]
: MYID TID-X ;
: SYNC BARRIER ;
MYID SYNC MYID CELLS DATA + !
