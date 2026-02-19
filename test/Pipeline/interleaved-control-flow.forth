\ RUN: %warpforth-translate --forth-to-mlir %s | %warpforth-opt --warpforth-pipeline | %FileCheck %s
\ RUN: %warpforth-translate --forth-to-mlir %s | %warpforth-opt --convert-forth-to-memref --convert-forth-to-gpu | %FileCheck %s --check-prefix=MID

\ Verify interleaved control flow (multi-WHILE, WHILE+UNTIL) compiles
\ through the full pipeline to gpu.binary.
\ CHECK: gpu.binary @warpforth_module

\ Verify intermediate MLIR: gpu.func with cf branches, no scf ops
\ MID: gpu.module @warpforth_module
\ MID: gpu.func @main(%arg0: memref<4xi64> {forth.param_name = "DATA"}) kernel
\ MID: gpu.return

\ Multi-WHILE: two cond_br exits + one unconditional back-edge
\ MID: func.func private @MULTI_WHILE
\ MID: cf.cond_br
\ MID: cf.cond_br
\ MID: cf.br

\ WHILE+UNTIL: WHILE exit + UNTIL exit merge at THEN
\ MID: func.func private @WHILE_UNTIL
\ MID: cf.cond_br
\ MID: cf.cond_br
\ MID: cf.br

\! kernel main
\! param DATA i64[4]
: multi-while
  BEGIN DUP 10 > WHILE DUP 2 MOD 0= WHILE 1 - REPEAT DROP THEN ;
: while-until
  BEGIN DUP 0 > WHILE 1 - DUP 5 = UNTIL THEN ;
multi-while while-until DATA 0 CELLS + !
