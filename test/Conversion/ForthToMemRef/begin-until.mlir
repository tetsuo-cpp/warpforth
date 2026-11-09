// RUN: %warpforth-opt --convert-forth-to-memref %s | %FileCheck %s

// Test: BEGIN...UNTIL loop conversion to memref with CF-based control flow
// Forth: 10 BEGIN 1 - DUP 0= UNTIL

// CHECK-LABEL: func.func private @main
// CHECK-NOT: memref<256xi64>

// CHECK: %[[N:.*]] = arith.constant 10 : i64
// CHECK: cf.br ^[[HEAD:bb[0-9]+]](%[[N]] : i64)
// CHECK: ^[[HEAD]](%[[CUR:.*]]: i64):
// CHECK: %[[ONE:.*]] = arith.constant 1 : i64
// CHECK: %[[DIFF:.*]] = arith.subi %[[CUR]], %[[ONE]] : i64
// CHECK: %[[ZERO:.*]] = arith.constant 0 : i64
// CHECK: %[[EQ:.*]] = arith.cmpi eq, %[[DIFF]], %[[ZERO]] : i64
// CHECK: %[[FLAG:.*]] = arith.extsi %[[EQ]] : i1 to i64
// CHECK: %[[Z2:.*]] = arith.constant 0 : i64
// CHECK: %[[COND:.*]] = arith.cmpi ne, %[[FLAG]], %[[Z2]] : i64
// CHECK: cf.cond_br %[[COND]], ^[[EXIT:bb[0-9]+]](%[[DIFF]] : i64), ^[[HEAD]](%[[DIFF]] : i64)
// CHECK: ^[[EXIT]](%{{.*}}: i64):
// CHECK: return
// CHECK-NOT: memref<256xi64>

module {
  func.func private @main() {
    %0 = forth.stack !forth.stack
    %1 = forth.constant %0(10 : i64) : !forth.stack -> !forth.stack
    cf.br ^bb1(%1 : !forth.stack)
  ^bb1(%2: !forth.stack):
    %3 = forth.constant %2(1 : i64) : !forth.stack -> !forth.stack
    %4 = forth.subi %3 : !forth.stack -> !forth.stack
    %5 = forth.dup %4 : !forth.stack -> !forth.stack
    %6 = forth.zero_eq %5 : !forth.stack -> !forth.stack
    %output_stack, %flag = forth.pop_flag %6 : !forth.stack -> !forth.stack, i1
    cf.cond_br %flag, ^bb2(%output_stack : !forth.stack), ^bb1(%output_stack : !forth.stack)
  ^bb2(%7: !forth.stack):
    return
  }
}
