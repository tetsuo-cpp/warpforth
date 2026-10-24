\ RUN: %not %warpforth-translate --forth-to-mlir %s 2>&1 | %FileCheck %s
\ CHECK: [[@LINE+2]]:6: error: unterminated comment: missing ')'
\! kernel main
: SQ ( n -- n*n DUP * ;
3 SQ
