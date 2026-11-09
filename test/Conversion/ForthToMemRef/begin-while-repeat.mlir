// RUN: %warpforth-opt --tag-forth-stack-depth --convert-forth-to-memref %s | %FileCheck %s

// Test: BEGIN...WHILE...REPEAT loop conversion to memref with CF-based control flow
// Forth: 10 BEGIN DUP 0 > WHILE 1 - REPEAT

// CHECK-LABEL: func.func private @main
// CHECK-NOT: memref<256xi64>

// CHECK: %[[N:.*]] = arith.constant 10 : i64
// CHECK: cf.br ^[[HEAD:bb[0-9]+]](%[[N]] : i64)
// CHECK: ^[[HEAD]](%[[CUR:.*]]: i64):
// CHECK: %[[ZERO:.*]] = arith.constant 0 : i64
// CHECK: %[[GT:.*]] = arith.cmpi sgt, %[[CUR]], %[[ZERO]] : i64
// CHECK: %[[FLAG:.*]] = arith.extsi %[[GT]] : i1 to i64
// CHECK: %[[Z2:.*]] = arith.constant 0 : i64
// CHECK: %[[COND:.*]] = arith.cmpi ne, %[[FLAG]], %[[Z2]] : i64
// CHECK: cf.cond_br %[[COND]], ^[[BODY:bb[0-9]+]](%[[CUR]] : i64), ^[[EXIT:bb[0-9]+]](%[[CUR]] : i64)
// CHECK: ^[[BODY]](%[[LEFT:.*]]: i64):
// CHECK: %[[ONE:.*]] = arith.constant 1 : i64
// CHECK: %[[DIFF:.*]] = arith.subi %[[LEFT]], %[[ONE]] : i64
// CHECK: cf.br ^[[HEAD]](%[[DIFF]] : i64)
// CHECK: ^[[EXIT]](%{{.*}}: i64):
// CHECK: return
// CHECK-NOT: memref<256xi64>

module {
  func.func private @main() {
    %0 = forth.stack !forth.stack
    %1 = forth.constant %0(10 : i64) : !forth.stack -> !forth.stack
    cf.br ^bb1(%1 : !forth.stack)
  ^bb1(%2: !forth.stack):
    %3 = forth.dup %2 : !forth.stack -> !forth.stack
    %4 = forth.constant %3(0 : i64) : !forth.stack -> !forth.stack
    %5 = forth.gti %4 : !forth.stack -> !forth.stack
    %output_stack, %flag = forth.pop_flag %5 : !forth.stack -> !forth.stack, i1
    cf.cond_br %flag, ^bb2(%output_stack : !forth.stack), ^bb3(%output_stack : !forth.stack)
  ^bb2(%6: !forth.stack):
    %7 = forth.constant %6(1 : i64) : !forth.stack -> !forth.stack
    %8 = forth.subi %7 : !forth.stack -> !forth.stack
    cf.br ^bb1(%8 : !forth.stack)
  ^bb3(%9: !forth.stack):
    return
  }
}
