\ RUN: %warpforthc -O0 %s -o %t.ptx
\ RUN: %FileCheck %s < %t.ptx

\ The loop carries its one cell in a register. Only the DO counter lives in
\ local memory, so even without optimization the local depot holds 8 bytes,
\ not a 2048-byte stack.

\! kernel main
\! param OUT i64[1]
5 10 2 DO I + LOOP OUT !

\ CHECK-LABEL: .entry main(
\ CHECK: .local .align 8 .b8 __local_depot0[8];
\ CHECK: st.global.u64
