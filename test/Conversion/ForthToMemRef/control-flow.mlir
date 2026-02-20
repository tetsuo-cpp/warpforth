// RUN: %warpforth-opt --convert-forth-to-memref %s | %FileCheck %s

// Test: IF/ELSE/THEN and IF/THEN conversion to memref with CF-based control flow
// Forth: 1 IF 42 ELSE 99 THEN  0 IF 7 THEN

// CHECK-LABEL: func.func private @main

// Stack allocation and literal 1 push:
// CHECK: %[[ALLOCA:.*]] = memref.alloca() : memref<256xi64>
// CHECK: %[[C1:.*]] = arith.constant 1 : i64
// CHECK: memref.store %[[C1]], %[[ALLOCA]]

// Pop flag and conditional branch:
// CHECK: %[[FLAG1:.*]] = memref.load
// CHECK: %[[ZERO1:.*]] = arith.constant 0 : i64
// CHECK: %[[COND1:.*]] = arith.cmpi ne, %[[FLAG1]], %[[ZERO1]] : i64
// CHECK: cf.cond_br %[[COND1]], ^bb1(%[[ALLOCA]], %{{.*}} : memref<256xi64>, index), ^bb2(%[[ALLOCA]], %{{.*}} : memref<256xi64>, index)

// Then branch: push 42
// CHECK: ^bb1(%{{.*}}: memref<256xi64>, %{{.*}}: index):
// CHECK: arith.constant 42 : i64
// CHECK: memref.store
// CHECK: cf.br ^bb3

// Else branch: push 99
// CHECK: ^bb2(%{{.*}}: memref<256xi64>, %{{.*}}: index):
// CHECK: arith.constant 99 : i64
// CHECK: memref.store
// CHECK: cf.br ^bb3

// Merge block: push 0, pop flag, second conditional branch
// CHECK: ^bb3(%{{.*}}: memref<256xi64>, %{{.*}}: index):
// CHECK: arith.constant 0 : i64
// CHECK: memref.store
// CHECK: arith.cmpi ne
// CHECK: cf.cond_br

// Second IF true branch: push 7
// CHECK: ^bb4(%{{.*}}: memref<256xi64>, %{{.*}}: index):
// CHECK: arith.constant 7 : i64
// CHECK: memref.store

// Final merge and return
// CHECK: ^bb5(%{{.*}}: memref<256xi64>, %{{.*}}: index):
// CHECK: return

module {
  func.func private @main() {
    %0 = forth.stack !forth.stack
    %1 = forth.constant %0(1 : i64) : !forth.stack -> !forth.stack
    %output_stack, %flag = forth.pop_flag %1 : !forth.stack -> !forth.stack, i1
    cf.cond_br %flag, ^bb1(%output_stack : !forth.stack), ^bb2(%output_stack : !forth.stack)
  ^bb1(%2: !forth.stack):
    %3 = forth.constant %2(42 : i64) : !forth.stack -> !forth.stack
    cf.br ^bb3(%3 : !forth.stack)
  ^bb2(%4: !forth.stack):
    %5 = forth.constant %4(99 : i64) : !forth.stack -> !forth.stack
    cf.br ^bb3(%5 : !forth.stack)
  ^bb3(%6: !forth.stack):
    %7 = forth.constant %6(0 : i64) : !forth.stack -> !forth.stack
    %output_stack_0, %flag_1 = forth.pop_flag %7 : !forth.stack -> !forth.stack, i1
    cf.cond_br %flag_1, ^bb4(%output_stack_0 : !forth.stack), ^bb5(%output_stack_0 : !forth.stack)
  ^bb4(%8: !forth.stack):
    %9 = forth.constant %8(7 : i64) : !forth.stack -> !forth.stack
    cf.br ^bb5(%9 : !forth.stack)
  ^bb5(%10: !forth.stack):
    return
  }
}
