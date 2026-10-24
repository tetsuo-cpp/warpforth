\ RUN: %warpforthc %s -o %t.ptx
\ RUN: %FileCheck %s < %t.ptx
\ RUN: %FileCheck %s --check-prefix=GENERIC < %t.ptx

\ Global memory words address the global state space even after address
\ arithmetic, so every indexed access is ld.global/st.global.

\ GENERIC-NOT: {{^[[:space:]]+(ld|st)\.[bsuf][0-9]+}}

\! kernel main
\! param A i64[256]
\! param F f64[256]
\! param C8 i64[256]
\! param C16 i64[256]
\! param C32 i64[256]
\! param H i64[256]
\! param BH i64[256]
\! param F4 i64[256]
\! shared S i64[256]

\ CHECK-LABEL: .entry main

\ CHECK: ld.global.{{[a-z]+}}64
\ CHECK: st.global.{{[a-z]+}}64
GLOBAL-ID CELLS A + @
GLOBAL-ID 1 + CELLS A + !

\ CHECK: ld.global.{{[a-z]+}}64
\ CHECK: st.global.{{[a-z]+}}64
GLOBAL-ID CELLS F + F@
GLOBAL-ID 1 + CELLS F + F!

\ CHECK: ld.global.{{[a-z]+}}8
\ CHECK: st.global.{{[a-z]+}}8
GLOBAL-ID C8 + I8@
GLOBAL-ID 1 + C8 + I8!

\ CHECK: ld.global.{{[a-z]+}}16
\ CHECK: st.global.{{[a-z]+}}16
GLOBAL-ID 2 * C16 + I16@
GLOBAL-ID 1 + 2 * C16 + I16!

\ CHECK: ld.global.{{[a-z]+}}32
\ CHECK: st.global.{{[a-z]+}}32
GLOBAL-ID 4 * C32 + I32@
GLOBAL-ID 1 + 4 * C32 + I32!

\ CHECK: ld.global.{{[a-z]+}}16
\ CHECK: st.global.{{[a-z]+}}16
GLOBAL-ID 2 * H + HF@
GLOBAL-ID 1 + 2 * H + HF!

\ CHECK: ld.global.{{[a-z]+}}16
\ CHECK: st.global.{{[a-z]+}}16
GLOBAL-ID 2 * BH + BF@
GLOBAL-ID 1 + 2 * BH + BF!

\ CHECK: ld.global.{{[a-z]+}}32
\ CHECK: st.global.{{[a-z]+}}32
GLOBAL-ID 4 * F4 + F32@
GLOBAL-ID 1 + 4 * F4 + F32!

\ Shared words keep the shared state space.
\ CHECK: ld.shared.{{[a-z]+}}64
\ CHECK: st.shared.{{[a-z]+}}64
TID-X CELLS S + S@
TID-X 1 + CELLS S + S!
