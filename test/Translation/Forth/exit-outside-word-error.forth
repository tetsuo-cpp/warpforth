\ RUN: %not %warpforth-translate --forth-to-mlir %s 2>&1 | %FileCheck %s
\ CHECK: {{.*}}exit-outside-word-error.forth:4:1: error: EXIT outside word definition
\! kernel main
EXIT
