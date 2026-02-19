// RUN: %warpforth-opt --convert-forth-to-memref %s | %FileCheck %s

// Test: DO...LOOP with I conversion to memref with CF-based control flow
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

// Loop header: load counter, compare < limit, cond_br
// CHECK: ^bb1(%{{.*}}: memref<256xi64>, %{{.*}}: index):
// CHECK: memref.load %[[COUNTER]]
// CHECK: arith.cmpi slt
// CHECK: cf.cond_br %{{.*}}, ^bb2(%{{.*}}: memref<256xi64>, index), ^bb3(%{{.*}}: memref<256xi64>, index)

// Loop body: push I (load counter, push to stack), increment counter
// CHECK: ^bb2(%{{.*}}: memref<256xi64>, %{{.*}}: index):
// CHECK: memref.load %[[COUNTER]]
// CHECK: memref.store
// CHECK: memref.load %[[COUNTER]]
// CHECK: arith.constant 1 : i64
// CHECK: arith.addi
// CHECK: memref.store %{{.*}}, %[[COUNTER]]
// CHECK: cf.br ^bb1

// Exit block
// CHECK: ^bb3(%{{.*}}: memref<256xi64>, %{{.*}}: index):
// CHECK: return

module {
  func.func private @main() {
    %0 = forth.stack !forth.stack
    %1 = forth.literal %0 10 : !forth.stack -> !forth.stack
    %2 = forth.literal %1 0 : !forth.stack -> !forth.stack
    %output_stack, %value = forth.pop %2 : !forth.stack -> !forth.stack, i64
    %output_stack_0, %value_1 = forth.pop %output_stack : !forth.stack -> !forth.stack, i64
    %alloca = memref.alloca() : memref<1xi64>
    %c0 = arith.constant 0 : index
    memref.store %value, %alloca[%c0] : memref<1xi64>
    cf.br ^bb1(%output_stack_0 : !forth.stack)
  ^bb1(%3: !forth.stack):
    %c0_2 = arith.constant 0 : index
    %4 = memref.load %alloca[%c0_2] : memref<1xi64>
    %5 = arith.cmpi slt, %4, %value_1 : i64
    cf.cond_br %5, ^bb2(%3 : !forth.stack), ^bb3(%3 : !forth.stack)
  ^bb2(%6: !forth.stack):
    %c0_3 = arith.constant 0 : index
    %7 = memref.load %alloca[%c0_3] : memref<1xi64>
    %8 = forth.push_value %6, %7 : !forth.stack, i64 -> !forth.stack
    %c0_4 = arith.constant 0 : index
    %9 = memref.load %alloca[%c0_4] : memref<1xi64>
    %c1_i64 = arith.constant 1 : i64
    %10 = arith.addi %9, %c1_i64 : i64
    memref.store %10, %alloca[%c0_4] : memref<1xi64>
    cf.br ^bb1(%8 : !forth.stack)
  ^bb3(%11: !forth.stack):
    return
  }
}
