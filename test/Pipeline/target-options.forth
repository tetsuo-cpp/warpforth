\ RUN: %warpforth-translate --forth-to-mlir %s | %warpforth-opt '--pass-pipeline=builtin.module(warpforth-pipeline{chip=sm_80 opt-level=3 compilation-target=isa})' --mlir-print-ir-after=nvvm-attach-target --mlir-disable-threading 2>&1 | %FileCheck %s
\ RUN: %warpforth-translate --forth-to-mlir %s | %not %warpforth-opt '--pass-pipeline=builtin.module(warpforth-pipeline{opt-level=4})' 2>&1 | %FileCheck %s --check-prefix=BAD-OPT
\ RUN: %warpforth-translate --forth-to-mlir %s | %not %warpforth-opt '--pass-pipeline=builtin.module(warpforth-pipeline{compilation-target=bogus})' 2>&1 | %FileCheck %s --check-prefix=BAD-FORMAT

\! kernel main
42

\ CHECK: #nvvm.target<O = 3, chip = "sm_80"
\ BAD-OPT: for the --opt-level option: Cannot find option named '4'!
\ BAD-FORMAT: for the --compilation-target option: Cannot find option named 'bogus'!
