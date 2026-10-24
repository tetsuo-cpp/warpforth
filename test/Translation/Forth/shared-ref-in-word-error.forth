\ RUN: %not %warpforth-translate --forth-to-mlir %s 2>&1 | %FileCheck %s
\ CHECK: shared memory 'SCRATCH' cannot be referenced inside a word definition
\ CHECK-NOT: unknown word
\! kernel main
\! shared SCRATCH i64[256]
: BAD-WORD SCRATCH @ ;
BAD-WORD
