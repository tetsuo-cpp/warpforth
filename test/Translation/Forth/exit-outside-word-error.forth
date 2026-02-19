\ RUN: %not %warpforth-translate --forth-to-mlir %s 2>&1 | %FileCheck %s
\ CHECK: EXIT outside word definition
\! kernel main
EXIT
