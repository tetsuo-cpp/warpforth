\ RUN: %not %warpforth-translate --forth-to-mlir %s 2>&1 | %FileCheck %s
\ CHECK: local variable name 'DATA' conflicts with parameter name
\! kernel main
\! param DATA i64[256]
: BAD { data -- } data ;
BAD
