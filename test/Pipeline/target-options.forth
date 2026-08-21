\ RUN: %warpforth-translate --forth-to-mlir %s | %warpforth-opt '--pass-pipeline=builtin.module(warpforth-pipeline{chip=sm_80 opt-level=3 compilation-target=isa})' --mlir-print-ir-after=nvvm-attach-target --mlir-disable-threading 2>&1 | %FileCheck %s
\ RUN: %warpforthc --arch sm_80 --mlir-print-ir-after-all --mlir-disable-threading %s -o %t.ptx 2>&1 | %FileCheck %s --check-prefix=ARCH
\ RUN: %not %warpforthc --arch sm_99 %s -o %t.ptx 2>&1 | %FileCheck %s --check-prefix=BAD-ARCH
\ RUN: %warpforth-translate --forth-to-mlir %s | %not %warpforth-opt '--pass-pipeline=builtin.module(warpforth-pipeline{chip=sm_99})' 2>&1 | %FileCheck %s --check-prefix=BAD-ARCH
\ RUN: %warpforth-translate --forth-to-mlir %s | %not %warpforth-opt '--pass-pipeline=builtin.module(warpforth-pipeline{opt-level=4})' 2>&1 | %FileCheck %s --check-prefix=BAD-OPT
\ RUN: %warpforth-translate --forth-to-mlir %s | %not %warpforth-opt '--pass-pipeline=builtin.module(warpforth-pipeline{compilation-target=bogus})' 2>&1 | %FileCheck %s --check-prefix=BAD-FORMAT

\! kernel main
42

\ CHECK: #nvvm.target<O = 3, chip = "sm_80"
\ ARCH: #nvvm.target<chip = "sm_80"
\ BAD-ARCH: unsupported GPU architecture 'sm_99'; supported architectures: sm_70, sm_75, sm_80, sm_86, sm_89, sm_90
\ BAD-OPT: for the --opt-level option: Cannot find option named '4'!
\ BAD-FORMAT: for the --compilation-target option: Cannot find option named 'bogus'!
