\ RUN: %not %warpforth-translate --forth-to-mlir %s 2>&1 | %FileCheck %s
\ CHECK: unclosed control flow (missing THEN, REPEAT, or UNTIL?)
\! kernel main
BEGIN 1 WHILE 42
