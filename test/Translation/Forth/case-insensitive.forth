\ RUN: %warpforth-translate --forth-to-mlir %s | %FileCheck %s

\ Verify case-insensitive parsing: mixed-case input produces correct ops
\ CHECK: forth.dup
\ CHECK: forth.drop
\ CHECK: forth.swap
\ CHECK: forth.add
\! kernel main
1 Dup DROP swap duP +
