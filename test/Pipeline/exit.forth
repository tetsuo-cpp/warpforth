\ RUN: %warpforth-translate --forth-to-mlir %s | %warpforth-opt --warpforth-pipeline | %FileCheck %s
\ RUN: %warpforth-translate --forth-to-mlir %s | %warpforth-opt --inline --lower-forth-stack --canonicalize | %FileCheck %s --check-prefix=MID --implicit-check-not="arith.constant true" --implicit-check-not="arith.constant 42"
\ CHECK: gpu.binary @warpforth_module

\ EXIT leaves the inlined word early: the kernel stores the 7 pushed before
\ EXIT, and the code after EXIT is gone.
\ MID-LABEL: func.func @main
\ MID: %[[SEVEN:.*]] = arith.constant 7 : i64
\ MID: llvm.store %[[SEVEN]], %{{.*}} : i64, !llvm.ptr<1>
\ MID-NEXT: return

\! kernel main
\! param DATA i64[4]
: DO-EXIT 7 1 IF EXIT THEN DROP 42 ;
DO-EXIT DATA 0 CELLS + !
