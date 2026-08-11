\ RUN: %not %warpforth-translate --forth-to-mlir %s 2>&1 | %FileCheck %s
\ CHECK: {{.*}}if-until-mismatch-error.forth:4:4: error: UNTIL without matching BEGIN
\! kernel main
IF UNTIL
