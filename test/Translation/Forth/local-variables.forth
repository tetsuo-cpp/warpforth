\ RUN: %warpforth-translate --forth-to-mlir %s | %FileCheck %s

\ Test basic local variable binding and reference.

\ CHECK: func.func private @ADD3(%arg0: !forth.stack) -> !forth.stack {
\ CHECK:   forth.pop %arg0 : !forth.stack -> !forth.stack, i64
\ CHECK:   forth.pop %{{.*}} : !forth.stack -> !forth.stack, i64
\ CHECK:   forth.pop %{{.*}} : !forth.stack -> !forth.stack, i64
\ CHECK:   forth.push_value %{{.*}}, %{{.*}} : !forth.stack, i64 -> !forth.stack
\ CHECK:   forth.push_value %{{.*}}, %{{.*}} : !forth.stack, i64 -> !forth.stack
\ CHECK:   forth.addi
\ CHECK:   forth.push_value %{{.*}}, %{{.*}} : !forth.stack, i64 -> !forth.stack
\ CHECK:   forth.addi
\ CHECK:   return

\ CHECK: func.func private @SWAP2(%arg0: !forth.stack) -> !forth.stack {
\ CHECK:   forth.pop %arg0 : !forth.stack -> !forth.stack, i64
\ CHECK:   forth.pop %{{.*}} : !forth.stack -> !forth.stack, i64
\ CHECK:   forth.push_value %{{.*}}, %{{.*}} : !forth.stack, i64 -> !forth.stack
\ CHECK:   forth.push_value %{{.*}}, %{{.*}} : !forth.stack, i64 -> !forth.stack
\ CHECK:   return

\! kernel main
: ADD3 { a b c -- } a b + c + ;
: SWAP2 { x y -- } y x ;
1 2 3 ADD3
10 20 SWAP2
