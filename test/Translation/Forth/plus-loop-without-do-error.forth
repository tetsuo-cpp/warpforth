\ RUN: %not %warpforth-translate --forth-to-mlir %s 2>&1 | %FileCheck %s
\ CHECK: +LOOP without matching DO
\! kernel main
+LOOP
