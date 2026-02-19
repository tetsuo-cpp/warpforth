\ RUN: %not %warpforth-translate --forth-to-mlir %s 2>&1 | %FileCheck %s
\ CHECK: duplicate \! kernel directive
\! kernel main
\! kernel other
