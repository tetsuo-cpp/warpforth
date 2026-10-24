\ RUN: %not %warpforth-translate --forth-to-mlir %s 2>&1 | %FileCheck %s
\ CHECK: [[@LINE+3]]:11: error: unterminated comment: missing ')'
\! kernel main
: SQ DUP * ;
3 SQ DROP ( never closed
1 2 +
