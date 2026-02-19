// RUN: %warpforth-opt --convert-forth-to-memref %s | %FileCheck %s

// Test: DO...LEAVE...LOOP lowers to CF branch to loop exit.
// Forth: 10 0 DO LEAVE LOOP

// CHECK-LABEL: func.func private @main
// CHECK: %[[STACK:.*]] = memref.alloca() : memref<256xi64>
// CHECK: cf.br ^bb1(%[[STACK]], %{{.*}} : memref<256xi64>, index)
// CHECK: ^bb1(%{{.*}}: memref<256xi64>, %{{.*}}: index):
// CHECK: cf.cond_br %{{.*}}, ^bb2(%{{.*}}: memref<256xi64>, index), ^bb3(%{{.*}}: memref<256xi64>, index)
// CHECK: ^bb2(%{{.*}}: memref<256xi64>, %{{.*}}: index):
// CHECK-NEXT: cf.br ^bb3(%{{.*}}: memref<256xi64>, index)
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
    cf.br ^bb3(%6 : !forth.stack)
  ^bb3(%7: !forth.stack):
    return
  }
}
