\ RUN: %warpforthc -O3 --features=+ptx70 --mlir-print-ir-after-all --mlir-disable-threading %s -o %t.ptx 2>&1 | %FileCheck %s
\ RUN: %warpforthc --opt-level=0 --arch sm_80 --mlir-print-ir-after-all --mlir-disable-threading %s -o %t.ptx 2>&1 | %FileCheck %s --check-prefix=O0
\ RUN: %warpforthc --features=+ptx70 %s -o - | %FileCheck %s --check-prefix=PTX
\ RUN: %not %warpforthc -O4 %s -o %t.ptx 2>&1 | %FileCheck %s --check-prefix=BAD-OPT

\! kernel main
42

\ CHECK: #nvvm.target<O = 3, chip = "sm_70", features = "+ptx70"
\ O0: #nvvm.target<O = 0, chip = "sm_80"
\ PTX: .version 7.0
\ BAD-OPT: Cannot find option named '4'!
