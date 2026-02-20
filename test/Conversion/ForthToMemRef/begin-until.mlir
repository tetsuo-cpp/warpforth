// RUN: %warpforth-opt --convert-forth-to-memref %s | %FileCheck %s

// Test: BEGIN...UNTIL loop conversion to memref with CF-based control flow
// Forth: 10 BEGIN 1 - DUP 0= UNTIL

// CHECK-LABEL: func.func private @main

// Stack allocation and literal 10 push:
// CHECK: %[[ALLOCA:.*]] = memref.alloca() : memref<256xi64>
// CHECK: %[[C10:.*]] = arith.constant 10 : i64
// CHECK: memref.store %[[C10]], %[[ALLOCA]]
// CHECK: cf.br ^bb1

// Loop body: push 1, subtract, dup, zero_eq, pop_flag, cond_br
// CHECK: ^bb1(%{{.*}}: memref<256xi64>, %{{.*}}: index):
// CHECK: arith.constant 1 : i64
// CHECK: memref.store
// CHECK: arith.subi
// CHECK: memref.store
// CHECK: memref.store
// CHECK: arith.cmpi eq
// CHECK: arith.extsi
// CHECK: memref.store
// CHECK: arith.cmpi ne
// CHECK: cf.cond_br %{{.*}}, ^bb2(%{{.*}}: memref<256xi64>, index), ^bb1(%{{.*}}: memref<256xi64>, index)

// Exit block
// CHECK: ^bb2(%{{.*}}: memref<256xi64>, %{{.*}}: index):
// CHECK: return

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
