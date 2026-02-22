\ RUN: %not %warpforth-translate --forth-to-mlir %s 2>&1 | %FileCheck %s
\ CHECK: local variable name 'BUF' conflicts with shared memory name
\! kernel main
\! shared BUF i32[64]
: BAD { buf -- } buf ;
BAD
