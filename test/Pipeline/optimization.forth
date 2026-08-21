\ RUN: %warpforth-translate --forth-to-mlir %s | %warpforth-opt --warpforth-pipeline --mlir-print-ir-after=inline --mlir-disable-threading 2>&1 | %FileCheck %s --check-prefix=INLINE
\ RUN: %warpforth-translate --forth-to-mlir %s | %warpforth-opt --warpforth-pipeline --mlir-print-ir-before=cse --mlir-print-ir-after=cse --mlir-disable-threading 2>&1 | %FileCheck %s --check-prefix=CSE

\ The default pipeline inlines the user word and removes its dead definition.
\ INLINE: IR Dump After Inliner (inline)
\ INLINE-LABEL: func.func @main(
\ INLINE-NOT: func.call
\ INLINE-NOT: func.func private @DOUBLE

\ CSE eliminates duplicate address extraction from repeated parameter refs.
\ CSE: IR Dump Before CSE (cse)
\ CSE-COUNT-2: memref.extract_aligned_pointer_as_index %arg0
\ CSE: IR Dump After CSE (cse)
\ CSE: memref.extract_aligned_pointer_as_index %arg0
\ CSE-NOT: memref.extract_aligned_pointer_as_index %arg0

\! kernel main
\! param DATA i64[4]
: DOUBLE DUP + ;
5 DOUBLE DROP
DATA DROP DATA DROP
