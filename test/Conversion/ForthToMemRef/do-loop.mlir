// RUN: %warpforth-opt --convert-forth-to-memref %s | %FileCheck %s

// Test: DO...LOOP with I conversion to memref with post-test crossing check
// Forth: 10 0 DO I LOOP

// CHECK-LABEL: func.func private @main

// Stack allocation and push 10, 0:
// CHECK: %[[ALLOCA:.*]] = memref.alloca() : memref<256xi64>
// CHECK: arith.constant 10 : i64
// CHECK: memref.store
// CHECK: arith.constant 0 : i64
// CHECK: memref.store

// Pop start and limit from stack:
// CHECK: memref.load
// CHECK: arith.subi
// CHECK: memref.load
// CHECK: arith.subi

// Loop counter alloca and initialization:
// CHECK: %[[COUNTER:.*]] = memref.alloca() : memref<1xi64>
// CHECK: memref.store %{{.*}}, %[[COUNTER]]
// CHECK: cf.br ^bb1

// Loop body: push I (load counter, push to stack), crossing test
// CHECK: ^bb1(%{{.*}}: memref<256xi64>, %{{.*}}: index):
// CHECK: memref.load %[[COUNTER]]
// CHECK: memref.store
// CHECK: arith.addi
// CHECK: memref.store %{{.*}}, %[[COUNTER]]
// CHECK: arith.subi
// CHECK: arith.subi
// CHECK: arith.xori
// CHECK: arith.cmpi slt
// CHECK: cf.cond_br

// Exit block
// CHECK: ^bb2(%{{.*}}: memref<256xi64>, %{{.*}}: index):
// CHECK: return

module {
  func.func private @main() {
    %0 = forth.stack !forth.stack
    %1 = forth.constant %0(10 : i64) : !forth.stack -> !forth.stack
    %2 = forth.constant %1(0 : i64) : !forth.stack -> !forth.stack
    %output_stack, %value = forth.pop %2 : !forth.stack -> !forth.stack, i64
    %output_stack_0, %value_1 = forth.pop %output_stack : !forth.stack -> !forth.stack, i64
    %alloca = memref.alloca() : memref<1xi64>
    %c0 = arith.constant 0 : index
    memref.store %value, %alloca[%c0] : memref<1xi64>
    cf.br ^bb1(%output_stack_0 : !forth.stack)
  ^bb1(%3: !forth.stack):
    %c0_2 = arith.constant 0 : index
    %4 = memref.load %alloca[%c0_2] : memref<1xi64>
    %5 = forth.push_value %3, %4 : !forth.stack, i64 -> !forth.stack
    %c1_i64 = arith.constant 1 : i64
    %c0_3 = arith.constant 0 : index
    %6 = memref.load %alloca[%c0_3] : memref<1xi64>
    %7 = arith.addi %6, %c1_i64 : i64
    memref.store %7, %alloca[%c0_3] : memref<1xi64>
    %8 = arith.subi %6, %value_1 : i64
    %9 = arith.subi %7, %value_1 : i64
    %10 = arith.xori %8, %9 : i64
    %c0_i64 = arith.constant 0 : i64
    %11 = arith.cmpi slt, %10, %c0_i64 : i64
    cf.cond_br %11, ^bb2(%5 : !forth.stack), ^bb1(%5 : !forth.stack)
  ^bb2(%12: !forth.stack):
    return
  }
}
