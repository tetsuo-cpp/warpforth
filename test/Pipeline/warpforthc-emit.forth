\ RUN: %warpforthc --emit=mlir %s | %FileCheck %s --check-prefix=MLIR
\ RUN: %warpforthc --emit=llvm %s | %FileCheck %s --check-prefix=LLVM
\ RUN: %warpforthc --emit=ptx %s | %FileCheck %s --check-prefix=PTX
\ RUN: %not %warpforthc --emit=cubin %s 2>&1 | %FileCheck %s --check-prefix=BAD

\! kernel main
42

\ MLIR: func.func @main
\ MLIR: forth.constant
\ MLIR-NOT: gpu.module
\ LLVM: gpu.module @warpforth_module
\ LLVM: #nvvm.target
\ LLVM-NOT: gpu.binary
\ PTX: .version
\ BAD: Cannot find option named 'cubin'!
