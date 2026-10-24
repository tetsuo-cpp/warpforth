\ RUN: %warpforthc %s -o %t.ptx
\ RUN: %FileCheck %s --implicit-check-not='{{(ld|st)[.a-z]*\.[bsuf]32}}' < %t.ptx

\ i64 global accesses are naturally aligned, so each is a single 64-bit memory
\ operation.

\! kernel main
\! param A i64[256]
\! param B f64[256]
: SUM3 + + ;
1
GLOBAL-ID CELLS A + @
GLOBAL-ID CELLS B + F@ F>S
SUM3
GLOBAL-ID CELLS A + !

\ CHECK-LABEL: .entry main(
\ CHECK: ld{{(\.global)?}}.{{[bsuf]}}64 {{%[a-z]+[0-9]+}}, [
\ CHECK: ld{{(\.global)?}}.{{[bsuf]}}64 {{%[a-z]+[0-9]+}}, [
\ CHECK: st{{(\.global)?}}.{{[bsuf]}}64 [
