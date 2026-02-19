// RUN: %warpforth-opt --convert-forth-to-memref %s | %FileCheck %s

// Test: BEGIN...WHILE...REPEAT loop conversion to memref with CF-based control flow
// Forth: 10 BEGIN DUP 0 > WHILE 1 - REPEAT

// CHECK-LABEL: func.func private @main

// Stack allocation and literal 10 push:
// CHECK: %[[ALLOCA:.*]] = memref.alloca() : memref<256xi64>
// CHECK: %[[C10:.*]] = arith.constant 10 : i64
// CHECK: memref.store %[[C10]], %[[ALLOCA]]
// CHECK: cf.br ^bb1

// Condition block: DUP, push 0, compare >, pop_flag, cond_br
// CHECK: ^bb1(%{{.*}}: memref<256xi64>, %{{.*}}: index):
// CHECK: memref.load
// CHECK: memref.store
// CHECK: arith.constant 0 : i64
// CHECK: memref.store
// CHECK: arith.cmpi sgt
// CHECK: arith.extsi
// CHECK: memref.store
// CHECK: arith.cmpi ne
// CHECK: cf.cond_br %{{.*}}, ^bb2(%{{.*}}: memref<256xi64>, index), ^bb3(%{{.*}}: memref<256xi64>, index)

// Body block: push 1, subtract, branch back to condition
// CHECK: ^bb2(%{{.*}}: memref<256xi64>, %{{.*}}: index):
// CHECK: arith.constant 1 : i64
// CHECK: memref.store
// CHECK: arith.subi
// CHECK: memref.store
// CHECK: cf.br ^bb1

// Exit block
// CHECK: ^bb3(%{{.*}}: memref<256xi64>, %{{.*}}: index):
// CHECK: return

module {
  func.func private @main() {
    %0 = forth.stack !forth.stack
    %1 = forth.literal %0 10 : !forth.stack -> !forth.stack
    cf.br ^bb1(%1 : !forth.stack)
  ^bb1(%2: !forth.stack):
    %3 = forth.dup %2 : !forth.stack -> !forth.stack
    %4 = forth.literal %3 0 : !forth.stack -> !forth.stack
    %5 = forth.gt %4 : !forth.stack -> !forth.stack
    %output_stack, %flag = forth.pop_flag %5 : !forth.stack -> !forth.stack, i1
    cf.cond_br %flag, ^bb2(%output_stack : !forth.stack), ^bb3(%output_stack : !forth.stack)
  ^bb2(%6: !forth.stack):
    %7 = forth.literal %6 1 : !forth.stack -> !forth.stack
    %8 = forth.sub %7 : !forth.stack -> !forth.stack
    cf.br ^bb1(%8 : !forth.stack)
  ^bb3(%9: !forth.stack):
    return
  }
}
