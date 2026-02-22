\ RUN: %not %warpforth-translate --forth-to-mlir %s 2>&1 | %FileCheck %s
\ CHECK: shared memory 'SCRATCH' cannot be referenced inside a word definition
\! kernel main
\! shared SCRATCH i32[256]
: BAD-WORD SCRATCH @ ;
BAD-WORD
