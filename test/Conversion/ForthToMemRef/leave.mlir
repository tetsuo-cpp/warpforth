// RUN: %warpforth-opt --convert-forth-to-memref %s | %FileCheck %s

// Test: DO...LEAVE...LOOP lowers to CF branch to loop exit.
// Forth: 10 0 DO LEAVE LOOP

// CHECK-LABEL: func.func private @main
// CHECK: %[[STACK:.*]] = memref.alloca() : memref<256xi64>
// CHECK: cf.br ^bb1(%[[STACK]], %{{.*}} : memref<256xi64>, index)
// CHECK: ^bb1(%{{.*}}: memref<256xi64>, %{{.*}}: index):
// CHECK: cf.cond_br %true, ^bb2(%{{.*}}: memref<256xi64>, index), ^bb3(%{{.*}}: memref<256xi64>, index)
// CHECK: ^bb2(%{{.*}}: memref<256xi64>, %{{.*}}: index):
// CHECK: return

module {
  func.func private @main() {
    %0 = forth.stack !forth.stack
    %1 = forth.constant %0(10 : i32) : !forth.stack -> !forth.stack
    %2 = forth.constant %1(0 : i32) : !forth.stack -> !forth.stack
    %output_stack, %value = forth.pop %2 : !forth.stack -> !forth.stack, i64
    %output_stack_0, %value_1 = forth.pop %output_stack : !forth.stack -> !forth.stack, i64
    %alloca = memref.alloca() : memref<1xi64>
    %c0 = arith.constant 0 : index
    memref.store %value, %alloca[%c0] : memref<1xi64>
    cf.br ^bb1(%output_stack_0 : !forth.stack)
  ^bb1(%3: !forth.stack):
    %true = arith.constant true
    cf.cond_br %true, ^bb2(%3 : !forth.stack), ^bb3(%3 : !forth.stack)
  ^bb2(%4: !forth.stack):
    return
  ^bb3(%5: !forth.stack):
    %c1_i64 = arith.constant 1 : i64
    %c0_2 = arith.constant 0 : index
    %6 = memref.load %alloca[%c0_2] : memref<1xi64>
    %7 = arith.addi %6, %c1_i64 : i64
    memref.store %7, %alloca[%c0_2] : memref<1xi64>
    %8 = arith.subi %6, %value_1 : i64
    %9 = arith.subi %7, %value_1 : i64
    %10 = arith.xori %8, %9 : i64
    %c0_i64 = arith.constant 0 : i64
    %11 = arith.cmpi slt, %10, %c0_i64 : i64
    cf.cond_br %11, ^bb2(%5 : !forth.stack), ^bb1(%5 : !forth.stack)
  }
}
