\ RUN: %warpforth-translate --forth-to-mlir %s | %FileCheck %s

\ Paren comment lines before and between header directives are skipped like
\ '\' comment lines.
\ CHECK: func.func @main(%arg0: memref<4xi64> {forth.param_name = "A"}) attributes {forth.kernel}
\ CHECK-NEXT: forth.stack
\ CHECK-NEXT: forth.param_ref
\ CHECK-NEXT: forth.loadi
\ CHECK-NEXT: return
( Kernel description that
  spans lines )
\! kernel main
( 999 between directives )
\! param A i64[4]
A @
