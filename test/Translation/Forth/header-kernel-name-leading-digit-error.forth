\ RUN: %not %warpforth-translate --forth-to-mlir %s 2>&1 | %FileCheck %s
\ CHECK: :[[@LINE+1]]:1: error: kernel name must be a valid identifier
\! kernel 1main
