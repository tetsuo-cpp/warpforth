\ RUN: %warpforthc %s -o %t.ptx
\ RUN: %FileCheck %s --implicit-check-not=.func --implicit-check-not=shfl.sync < %t.ptx
\ RUN: %FileCheck %s --check-prefix=LANE < %t.ptx

\ CHECK-LABEL: .entry main
\ CHECK-COUNT-2: shfl.sync.down.b32 {{.*}}, -1;
\ CHECK-COUNT-2: shfl.sync.up.b32 {{.*}}, -1;
\ CHECK-COUNT-2: shfl.sync.bfly.b32 {{.*}}, -1;
\ CHECK-COUNT-2: shfl.sync.idx.b32 {{.*}}, -1;

\ LANE: mov.u32 %r{{[0-9]+}}, %laneid;

\! kernel main
\! param DATA i64[256]
: BFLY SHFL-XOR ;
GLOBAL-ID CELLS DATA + @
1 SHFL-DOWN 2 SHFL-UP 3 BFLY 4 SHFL-IDX
LANE-ID +
GLOBAL-ID CELLS DATA + !
