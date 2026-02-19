\ RUN: %not %warpforth-translate --forth-to-mlir %s 2>&1 | %FileCheck %s
\ CHECK: parameter 'DATA' cannot be referenced inside a word definition
\! kernel main
\! param DATA i64[256]
: BAD-WORD DATA @ ;
BAD-WORD
