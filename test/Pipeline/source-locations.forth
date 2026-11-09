\ RUN: %warpforth-translate --forth-to-mlir --mlir-print-debuginfo %s | %FileCheck %s --check-prefix=FORTH
\ RUN: %warpforthc --mlir-print-ir-after=lower-forth-stack --mlir-print-debuginfo --mlir-disable-threading %s -o %t.ptx 2>&1 | %FileCheck %s --check-prefix=LOWERED

\! kernel main
\! param OUT i64[1]
42 OUT !

\ FORTH: forth.constant %{{.*}}(42 : i64) {{.*}} loc([[FORTH_LOC:#loc[0-9]+]])
\ FORTH: [[FORTH_LOC]] = loc("{{.*}}source-locations.forth":6:1)

\ LOWERED: IR Dump After LowerForthStack (lower-forth-stack)
\ LOWERED: arith.constant 42 : i64 loc([[LOWERED_LOC:#loc[0-9]+]])
\ LOWERED: [[LOWERED_LOC]] = loc("{{.*}}source-locations.forth":6:1)
