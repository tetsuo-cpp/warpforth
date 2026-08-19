\ RUN: %warpforth-translate --forth-to-mlir --mlir-print-debuginfo %s | %FileCheck %s --check-prefix=FORTH
\ RUN: %warpforthc --mlir-print-ir-after=convert-forth-to-memref --mlir-print-debuginfo --mlir-disable-threading %s -o %t.ptx 2>&1 | %FileCheck %s --check-prefix=LOWERED

\! kernel main
42

\ FORTH: forth.constant %{{.*}}(42 : i64) {{.*}} loc([[FORTH_LOC:#loc[0-9]+]])
\ FORTH: [[FORTH_LOC]] = loc("{{.*}}source-locations.forth":5:1)

\ LOWERED: IR Dump After ConvertForthToMemRef (convert-forth-to-memref)
\ LOWERED: arith.constant 42 : i64 loc([[LOWERED_LOC:#loc[0-9]+]])
\ LOWERED: [[LOWERED_LOC]] = loc("{{.*}}source-locations.forth":5:1)
