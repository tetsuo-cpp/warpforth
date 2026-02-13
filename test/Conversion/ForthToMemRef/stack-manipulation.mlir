// RUN: %warpforth-opt --convert-forth-to-memref %s | %FileCheck %s

// CHECK-LABEL: func.func private @main

// dup: load top, increment SP, store copy
// CHECK: memref.load %{{.*}}[%{{.*}}] : memref<256xi64>
// CHECK: arith.addi
// CHECK: memref.store

// drop: decrement SP
// CHECK: arith.subi %{{.*}}, %{{.*}} : index

// swap: load top and second, store crossed
// CHECK: %[[TOP_VAL:.*]] = memref.load %{{.*}}[%[[SWAP_SP:.*]]] : memref<256xi64>
// CHECK: %[[SWAP_SP1:.*]] = arith.subi
// CHECK: %[[SEC_VAL:.*]] = memref.load %{{.*}}[%[[SWAP_SP1]]] : memref<256xi64>
// CHECK: memref.store %[[SEC_VAL]], %{{.*}}[%[[SWAP_SP]]] : memref<256xi64>
// CHECK: memref.store %[[TOP_VAL]], %{{.*}}[%[[SWAP_SP1]]] : memref<256xi64>

// over: load second element, increment SP, store copy
// CHECK: arith.subi
// CHECK: memref.load %{{.*}}[%{{.*}}] : memref<256xi64>
// CHECK: arith.addi
// CHECK: memref.store

// rot: load three values, store rotated (a b c -- b c a)
// CHECK: %[[ROT_C:.*]] = memref.load %{{.*}}[%[[ROT_SP:.*]]] : memref<256xi64>
// CHECK: %[[ROT_SP1:.*]] = arith.subi %[[ROT_SP]]
// CHECK: %[[ROT_B:.*]] = memref.load %{{.*}}[%[[ROT_SP1]]] : memref<256xi64>
// CHECK: %[[ROT_SP2:.*]] = arith.subi %[[ROT_SP]]
// CHECK: %[[ROT_A:.*]] = memref.load %{{.*}}[%[[ROT_SP2]]] : memref<256xi64>
// CHECK: memref.store %[[ROT_B]], %{{.*}}[%[[ROT_SP2]]] : memref<256xi64>
// CHECK: memref.store %[[ROT_C]], %{{.*}}[%[[ROT_SP1]]] : memref<256xi64>
// CHECK: memref.store %[[ROT_A]], %{{.*}}[%[[ROT_SP]]] : memref<256xi64>

module {
  func.func private @main() {
    %0 = forth.stack !forth.stack
    %1 = forth.literal %0 1 : !forth.stack -> !forth.stack
    %2 = forth.literal %1 2 : !forth.stack -> !forth.stack
    %3 = forth.literal %2 3 : !forth.stack -> !forth.stack
    %4 = forth.dup %3 : !forth.stack -> !forth.stack
    %5 = forth.drop %4 : !forth.stack -> !forth.stack
    %6 = forth.swap %5 : !forth.stack -> !forth.stack
    %7 = forth.over %6 : !forth.stack -> !forth.stack
    %8 = forth.rot %7 : !forth.stack -> !forth.stack
    return
  }
}
