\ RUN: %warpforth-translate --forth-to-mlir %s | %warpforth-opt --inline --lower-forth-stack | %FileCheck %s --check-prefix=MID
\ RUN: %warpforthc -O0 %s -o %t.ptx
\ RUN: %FileCheck %s --check-prefix=PTX < %t.ptx

\ Every cell's depth is known at compile time, so the sum is stored straight
\ from a register. Even without optimization the kernel reserves no local
\ memory for a data stack.

\! kernel main
\! param OUT i64[1]
2 3 + OUT !

\ MID-LABEL: func.func @main(
\ MID-NOT: memref.alloca
\ MID: %[[SUM:.*]] = arith.addi
\ MID-NOT: memref.alloca
\ MID: llvm.store %[[SUM]], %{{.*}} : i64, !llvm.ptr<1>
\ MID-NEXT: return

\ PTX-LABEL: .entry main(
\ PTX-NOT: __local_depot
\ PTX: st.global.u64
